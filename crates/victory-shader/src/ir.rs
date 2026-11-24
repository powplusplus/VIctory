//! One-to-one SSA form of a native program (decision 0003, step 1).
//!
//! Every machine instruction becomes one [`Inst`], except scheduling-only
//! ones (`s_nop`, `s_waitcnt*`, `s_inst_prefetch`...), which are dropped.
//! Each register dword is a variable: a read names the [`Value`] it sees, a
//! write defines a new one, and [`Phi`]s merge values where control flow
//! joins. Nothing is raised: an instruction keeps its opcode and decoded
//! fields, so its exact semantics stay with it.
//!
//! Implicit operands are explicit: SCC, M0, and EXEC, which every vector
//! instruction reads. A write to a VGPR also reads the register's old value,
//! because lanes disabled in EXEC keep it.
//!
//! M0-relative VGPR access (`v_movrels/d/sd_b32`, `v_swaprel_b32`) names its
//! operand's whole possible range: from the encoded register up to the
//! program's VGPR count, all read (and all written, for a destination).

use std::collections::HashMap;

use crate::ShaderError;
use crate::gfx10::{self, Fields, Flow, Instruction, NULL, Operand};

/// One register dword. Scalar codes as in [`Operand::Scalar`]: 0-105 SGPRs,
/// 106/107 VCC, 108-123 trap temporaries, 124 M0, 126/127 EXEC.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Reg {
    S(u8),
    Scc,
    V(u8),
}

pub const VCC: u8 = 106;
pub const M0: u8 = 124;
pub const EXEC: u8 = 126;

/// An SSA value: the content of one register dword (SCC: one bit).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Value(pub u32);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Program {
    pub wave64: bool,
    /// VGPRs v0 up to this exist; bounds M0-relative access.
    pub vgprs: u16,
    /// In code order, then copies made to give every loop one entry; block 0
    /// is the entry.
    pub blocks: Vec<Block>,
    /// Registers read before any write, with their value on entry: what the
    /// hardware or a called fetch routine set up (user SGPRs, system values,
    /// vertex inputs).
    pub inputs: Vec<(Reg, Value)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {
    /// Byte offset of the block's first machine instruction (shared by a
    /// block and its copies).
    pub offset: usize,
    pub preds: Vec<usize>,
    pub phis: Vec<Phi>,
    pub insts: Vec<Inst>,
    pub exit: Exit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exit {
    Jump(usize),
    /// `s_cbranch_*`, the block's last instruction.
    Branch {
        taken: usize,
        not_taken: usize,
    },
    /// `s_endpgm` and its variants.
    End,
    /// `s_setpc_b64` to an address held in registers, e.g. the handover to
    /// the next half of a merged hardware stage.
    Indirect,
}

impl Exit {
    pub fn successors(&self) -> Vec<usize> {
        match *self {
            Exit::Jump(b) => vec![b],
            Exit::Branch { taken, not_taken } => vec![taken, not_taken],
            Exit::End | Exit::Indirect => vec![],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Phi {
    pub reg: Reg,
    pub value: Value,
    /// One per predecessor, in [`Block::preds`] order.
    pub args: Vec<Value>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Inst {
    pub machine: Instruction,
    pub fields: Fields,
    /// Every register dword read, with the value it holds.
    pub reads: Vec<(Reg, Value)>,
    /// Every register dword written, with the value it receives.
    pub writes: Vec<(Reg, Value)>,
}

impl Inst {
    pub fn name(&self) -> &'static str {
        self.machine.name().unwrap_or("")
    }

    /// The value `reg` holds when this instruction reads it.
    pub fn read(&self, reg: Reg) -> Option<Value> {
        self.reads.iter().find(|r| r.0 == reg).map(|r| r.1)
    }

    /// The value this instruction writes to `reg`.
    pub fn write(&self, reg: Reg) -> Option<Value> {
        self.writes.iter().find(|w| w.0 == reg).map(|w| w.1)
    }
}

/// Decodes `code` and builds its SSA form; `vgprs` is the program's VGPR
/// count. Fails on instructions the IR cannot express yet
/// ([`ShaderError::Unsupported`]).
pub fn build(code: &[u8], wave64: bool, vgprs: u16) -> Result<Program, ShaderError> {
    let list = gfx10::decode(code)?;
    let lanes = 1 + u8::from(wave64);
    let isa = |offset, reason| ShaderError::Isa { offset, reason };

    let mut flows = Vec::with_capacity(list.len());
    let mut leaders = vec![0];
    for (i, inst) in list.iter().enumerate() {
        let flow = gfx10::control(inst, code.len(), |at| {
            list[..list.partition_point(|i| i.offset < at)].last()
        })?;
        if let Flow::Jump(t) | Flow::Branch(t) = flow {
            leaders.push(t);
        }
        if flow != Flow::Next
            && let Some(next) = list.get(i + 1)
        {
            leaders.push(next.offset);
        }
        flows.push(flow);
    }
    leaders.sort_unstable();
    leaders.dedup();
    let block_at = |offset: usize| {
        leaders
            .binary_search(&offset)
            .map_err(|_| isa(offset, "jump into a block"))
    };

    let mut blocks: Vec<Block> = leaders
        .iter()
        .map(|&offset| Block {
            offset,
            preds: Vec::new(),
            phis: Vec::new(),
            insts: Vec::new(),
            exit: Exit::End,
        })
        .collect();
    let next_offsets: Vec<Option<usize>> = (0..list.len())
        .map(|i| list.get(i + 1).map(|n| n.offset))
        .collect();
    for ((machine, flow), next) in list.into_iter().zip(flows).zip(next_offsets) {
        let b = leaders.partition_point(|&l| l <= machine.offset) - 1;
        let name = machine.name().ok_or(ShaderError::Unsupported {
            offset: machine.offset,
            what: "unknown opcode",
        })?;
        let fields = machine.fields(wave64);
        let fallthrough = || block_at(next.ok_or(isa(machine.offset, "falls off the code"))?);
        let ends = flow != Flow::Next || next.is_none_or(|n| leaders.binary_search(&n).is_ok());
        if ends {
            blocks[b].exit = match flow {
                Flow::Next => Exit::Jump(fallthrough()?),
                Flow::Jump(t) => Exit::Jump(block_at(t)?),
                Flow::Branch(t) => Exit::Branch {
                    taken: block_at(t)?,
                    not_taken: fallthrough()?,
                },
                Flow::Stop if name == "s_setpc_b64" => Exit::Indirect,
                Flow::Stop => Exit::End,
            };
        }
        if let Some((reads, writes)) = effects(&machine, &fields, name, lanes, vgprs)? {
            let unset = |regs: Vec<Reg>| regs.into_iter().map(|r| (r, NONE)).collect();
            blocks[b].insts.push(Inst {
                machine,
                fields,
                reads: unset(reads),
                writes: unset(writes),
            });
        }
    }
    link(&mut blocks);
    if !blocks[0].preds.is_empty() {
        return Err(ShaderError::Unsupported {
            offset: 0,
            what: "a branch to the program entry",
        });
    }
    split_irreducible(&mut blocks)?;
    thread_jumps(&mut blocks, wave64)?;

    let order = reverse_postorder(&blocks);
    let (live_in, live_out) = liveness(&blocks, &order);
    let mut values = 0u32;
    let mut new_value = || {
        values += 1;
        Value(values - 1)
    };
    let mut inputs = Vec::new();
    let mut outs: Vec<Vec<(Reg, Value)>> = vec![Vec::new(); blocks.len()];
    let mut current = [NONE; REGS];
    for &b in &order {
        current.fill(NONE);
        let block = &mut blocks[b];
        for reg in members(&live_in[b]) {
            current[index(reg)] = match block.preds[..] {
                [] => {
                    let v = new_value();
                    inputs.push((reg, v));
                    v
                }
                // A single predecessor comes earlier in reverse postorder.
                [p] => lookup(&outs[p], reg),
                _ => {
                    let value = new_value();
                    block.phis.push(Phi {
                        reg,
                        value,
                        args: Vec::new(),
                    });
                    value
                }
            };
        }
        for inst in &mut block.insts {
            for (reg, v) in &mut inst.reads {
                *v = current[index(*reg)];
                debug_assert_ne!(*v, NONE, "{reg:?} read at {:#x}", inst.machine.offset);
            }
            for (reg, v) in &mut inst.writes {
                *v = new_value();
                current[index(*reg)] = *v;
            }
        }
        outs[b] = members(&live_out[b])
            .map(|reg| (reg, current[index(reg)]))
            .collect();
    }
    for block in &mut blocks {
        for phi in &mut block.phis {
            phi.args = block
                .preds
                .iter()
                .map(|&p| lookup(&outs[p], phi.reg))
                .collect();
            debug_assert!(!phi.args.contains(&NONE), "{phi:?}");
        }
    }
    remove_trivial_phis(&mut blocks, values);
    Ok(Program {
        wave64,
        vgprs,
        blocks,
        inputs,
    })
}

/// Placeholder before renaming.
const NONE: Value = Value(u32::MAX);
/// SGPR codes 0-127, SCC, VGPRs 0-255.
const REGS: usize = 128 + 1 + 256;
type RegSet = [u64; REGS.div_ceil(64)];

fn index(reg: Reg) -> usize {
    match reg {
        Reg::S(c) => usize::from(c),
        Reg::Scc => 128,
        Reg::V(r) => 129 + usize::from(r),
    }
}

fn reg_at(i: usize) -> Reg {
    match i {
        0..128 => Reg::S(i as u8),
        128 => Reg::Scc,
        _ => Reg::V((i - 129) as u8),
    }
}

fn insert(set: &mut RegSet, reg: Reg) {
    let i = index(reg);
    set[i / 64] |= 1 << (i % 64);
}

fn remove(set: &mut RegSet, reg: Reg) {
    let i = index(reg);
    set[i / 64] &= !(1 << (i % 64));
}

fn contains(set: &RegSet, reg: Reg) -> bool {
    let i = index(reg);
    set[i / 64] >> (i % 64) & 1 != 0
}

/// In [`Reg`] order, which is also index order.
fn members(set: &RegSet) -> impl Iterator<Item = Reg> + '_ {
    (0..REGS)
        .filter(|&i| set[i / 64] >> (i % 64) & 1 != 0)
        .map(reg_at)
}

/// `outs` is sorted by register, and holds every register live out.
fn lookup(outs: &[(Reg, Value)], reg: Reg) -> Value {
    outs.binary_search_by_key(&reg, |&(r, _)| r)
        .map_or(NONE, |i| outs[i].1)
}

pub(crate) fn reverse_postorder(blocks: &[Block]) -> Vec<usize> {
    let mut seen = vec![false; blocks.len()];
    let mut post = Vec::with_capacity(blocks.len());
    let mut stack = vec![(0, 0)];
    seen[0] = true;
    while let Some(top) = stack.last_mut() {
        let (b, i) = *top;
        top.1 += 1;
        match blocks[b].exit.successors().get(i) {
            Some(&s) if !seen[s] => {
                seen[s] = true;
                stack.push((s, 0));
            }
            Some(_) => {}
            None => {
                post.push(b);
                stack.pop();
            }
        }
    }
    post.reverse();
    post
}

/// Recomputes every block's predecessors from the exits.
fn link(blocks: &mut [Block]) {
    for block in blocks.iter_mut() {
        block.preds.clear();
    }
    for b in 0..blocks.len() {
        for s in blocks[b].exit.successors() {
            blocks[s].preds.push(b);
        }
    }
}

/// An edge to an earlier block in reverse postorder whose target does not
/// dominate its source, if any: there the control flow is irreducible.
/// `index` is each block's position in `order`.
pub(crate) fn irreducible_edge(
    blocks: &[Block],
    order: &[usize],
    index: &[usize],
) -> Option<(usize, usize)> {
    let &entry = order.first()?;
    let idom = idoms(blocks, order, index);
    for &b in order {
        for s in blocks[b].exit.successors() {
            if index[s] > index[b] {
                continue;
            }
            let mut d = b;
            while d != s && d != entry {
                d = idom[d];
            }
            if d != s {
                return Some((b, s));
            }
        }
    }
    None
}

/// Each block's immediate dominator (the entry's is itself, unreachable
/// blocks' `usize::MAX`); `order` is reverse postorder, `index` each
/// block's position in it.
pub(crate) fn idoms(blocks: &[Block], order: &[usize], index: &[usize]) -> Vec<usize> {
    let mut idom = vec![usize::MAX; blocks.len()];
    let Some(&entry) = order.first() else {
        return idom;
    };
    idom[entry] = entry;
    let intersect = |idom: &[usize], mut a: usize, mut b: usize| {
        while a != b {
            while index[a] > index[b] {
                a = idom[a];
            }
            while index[b] > index[a] {
                b = idom[b];
            }
        }
        a
    };
    let mut changed = true;
    while changed {
        changed = false;
        for &b in &order[1..] {
            let mut new = usize::MAX;
            for &q in blocks[b].preds.iter().filter(|&&q| idom[q] != usize::MAX) {
                new = if new == usize::MAX {
                    q
                } else {
                    intersect(&idom, q, new)
                };
            }
            if idom[b] != new {
                idom[b] = new;
                changed = true;
            }
        }
    }
    idom
}

/// Node splitting, before SSA. While a loop is entered at several blocks
/// (a compiler-rotated loop whose peeled first test jumps into its body),
/// the entry first in reverse postorder stays the header, and another
/// entry's outside predecessors move to a copy of everything that entry
/// reaches in the loop before the header. The copies run the same
/// instructions, so behaviour is unchanged.
fn split_irreducible(blocks: &mut Vec<Block>) -> Result<(), ShaderError> {
    let limit = 4 * blocks.len() + 64;
    loop {
        let order = reverse_postorder(blocks);
        let mut index = vec![usize::MAX; blocks.len()];
        for (i, &b) in order.iter().enumerate() {
            index[b] = i;
        }
        let Some((from, _)) = irreducible_edge(blocks, &order, &index) else {
            return Ok(());
        };
        let found = multi_entry_loop(blocks, &vec![true; blocks.len()]);
        let Some((region, mut entries)) = found.filter(|_| blocks.len() < limit) else {
            return Err(ShaderError::Unsupported {
                offset: blocks[from].offset,
                what: "irreducible control flow",
            });
        };
        entries.sort_by_key(|&e| index[e]);
        let (header, entry) = (entries[0], entries[1]);

        // What `entry` reaches in the loop before the header, and its copies.
        let mut copy = vec![usize::MAX; blocks.len()];
        let mut copied = Vec::new();
        let mut stack = vec![entry];
        while let Some(b) = stack.pop() {
            if copy[b] != usize::MAX {
                continue;
            }
            copy[b] = blocks.len() + copied.len();
            copied.push(b);
            for s in blocks[b].exit.successors() {
                if region[s] && s != header {
                    stack.push(s);
                }
            }
        }
        let retarget = |exit: &mut Exit, to: &dyn Fn(usize) -> usize| match exit {
            Exit::Jump(s) => *s = to(*s),
            Exit::Branch { taken, not_taken } => {
                *taken = to(*taken);
                *not_taken = to(*not_taken);
            }
            Exit::End | Exit::Indirect => {}
        };
        for &b in &copied {
            let mut block = blocks[b].clone();
            retarget(&mut block.exit, &|s| {
                if copy[s] == usize::MAX { s } else { copy[s] }
            });
            blocks.push(block);
        }
        let outside: Vec<usize> = blocks[entry]
            .preds
            .iter()
            .copied()
            .filter(|&p| !region[p])
            .collect();
        for p in outside {
            retarget(&mut blocks[p].exit, &|s| {
                if s == entry { copy[entry] } else { s }
            });
        }
        link(blocks);
    }
}

/// A loop among the `inside` blocks entered at more than one block, with
/// its entries. Loops nested in a single-entry loop are searched with its
/// header removed, so one is found whenever the control flow is irreducible.
fn multi_entry_loop(blocks: &[Block], inside: &[bool]) -> Option<(Vec<bool>, Vec<usize>)> {
    let reach = |from: usize, forward: bool| {
        let mut seen = vec![false; blocks.len()];
        let mut stack = vec![from];
        while let Some(b) = stack.pop() {
            let next = if forward {
                blocks[b].exit.successors()
            } else {
                blocks[b].preds.clone()
            };
            for n in next {
                if inside[n] && !seen[n] {
                    seen[n] = true;
                    stack.push(n);
                }
            }
        }
        seen
    };
    let mut done = vec![false; blocks.len()];
    for v in (0..blocks.len()).filter(|&v| inside[v]) {
        if done[v] {
            continue;
        }
        // Blocks on a cycle through `v`; `v` itself only if there is one.
        let (ahead, behind) = (reach(v, true), reach(v, false));
        let region: Vec<bool> = ahead.iter().zip(&behind).map(|(a, b)| a & b).collect();
        if !region[v] {
            continue;
        }
        let members = (0..blocks.len()).filter(|&b| region[b]);
        for b in members.clone() {
            done[b] = true;
        }
        let entries: Vec<usize> = members
            .filter(|&b| blocks[b].preds.iter().any(|&p| !region[p]))
            .collect();
        match entries[..] {
            [] => {}
            [header] => {
                let mut nested = region;
                nested[header] = false;
                if let Some(found) = multi_entry_loop(blocks, &nested) {
                    return Some(found);
                }
            }
            _ => return Some((region, entries)),
        }
    }
    None
}

/// Correlated-branch jump threading, before SSA.
///
/// The console compiler often skips a conditional load block after setting a
/// sentinel flag (`v_mov_b32 v, 0; s_cbranch_scc1 merge`), then tests that
/// flag at the join block before branching around uses of the loaded SGPRs.
/// On the skipped edge those SGPRs still hold dead values from earlier code
/// (e.g. a descriptor or lane mask), which would form spurious mixed phis
/// unless the statically known edge is threaded past the join's branch.
fn thread_jumps(blocks: &mut Vec<Block>, wave64: bool) -> Result<(), ShaderError> {
    const MAX_JOIN_INSTS: usize = 32;
    const MAX_CHAIN: usize = 4;
    let limit = blocks.len() + blocks.len().max(64);
    let empty_consts = HashMap::new();
    let mut threaded = false;

    for _ in 0..8 {
        let order = reverse_postorder(blocks);
        let mut index = vec![usize::MAX; blocks.len()];
        for (i, &b) in order.iter().enumerate() {
            index[b] = i;
        }
        let mut live_in: Option<Vec<RegSet>> = None;
        let mut changed = false;

        let count = blocks.len();
        for m in 0..count {
            if blocks.len() >= limit {
                break;
            }
            if index[m] == usize::MAX || blocks[m].preds.len() < 2 {
                continue;
            }
            let Exit::Branch { taken, not_taken } = blocks[m].exit else {
                continue;
            };
            if taken == not_taken || blocks[m].insts.len() > MAX_JOIN_INSTS {
                continue;
            }
            if !is_threadable_join(&blocks[m]) {
                continue;
            }
            // Only thread branches that depend on an incoming edge's constants
            // (not ones that trivially fold without them, e.g. testing EXEC).
            if eval_join_branch(&blocks[m], &empty_consts, wave64).is_some() {
                continue;
            }
            let preds = blocks[m].preds.clone();
            for p in preds {
                if blocks.len() >= limit {
                    break;
                }
                if p == m
                    || index[p] == usize::MAX
                    || index[p] >= index[m]
                    || !blocks[p].exit.successors().contains(&m)
                {
                    continue;
                }
                let consts = incoming_constants(blocks, p, MAX_CHAIN);
                if consts.is_empty() {
                    continue;
                }
                let Some(go_taken) = eval_join_branch(&blocks[m], &consts, wave64) else {
                    continue;
                };
                let target = if go_taken { taken } else { not_taken };
                if target == m || target == 0 {
                    continue;
                }

                let live = live_in.get_or_insert_with(|| liveness(blocks, &order).0);
                let pruned = prune_join_body(&blocks[m], &live[target]);
                let dest = if pruned.is_empty() {
                    target
                } else {
                    let copy_idx = blocks.len();
                    blocks.push(Block {
                        offset: blocks[m].offset,
                        preds: Vec::new(),
                        phis: Vec::new(),
                        insts: pruned,
                        exit: Exit::Jump(target),
                    });
                    copy_idx
                };

                retarget_pred_edge(&mut blocks[p], m, dest);
                changed = true;
                threaded = true;
            }
        }

        if !changed {
            break;
        }
        prune_unreachable(blocks);
        link(blocks);
    }

    if threaded {
        split_irreducible(blocks)?;
    }
    Ok(())
}

/// Clears unreachable blocks so `link` never leaves an unreachable predecessor
/// in a reachable block's `preds` list.
fn prune_unreachable(blocks: &mut [Block]) {
    let order = reverse_postorder(blocks);
    let mut reachable = vec![false; blocks.len()];
    for &b in &order {
        reachable[b] = true;
    }
    for (b, block) in blocks.iter_mut().enumerate() {
        if !reachable[b] {
            block.insts.clear();
            block.exit = Exit::End;
        }
    }
}

fn retarget_pred_edge(pred: &mut Block, from: usize, to: usize) {
    match &mut pred.exit {
        Exit::Jump(s) if *s == from => *s = to,
        Exit::Branch { taken, not_taken } => {
            if *taken == from {
                *taken = to;
            }
            if *not_taken == from {
                *not_taken = to;
            }
            if *taken == *not_taken {
                let target = *taken;
                pred.exit = Exit::Jump(target);
                if pred
                    .insts
                    .last()
                    .is_some_and(|i| i.name().starts_with("s_cbranch_"))
                {
                    pred.insts.pop();
                }
            }
        }
        _ => {}
    }
}

/// Whether `block` is a small, side-effect-free, EXEC-preserving conditional
/// join whose body can safely be evaluated and duplicated on a threaded edge.
fn is_threadable_join(block: &Block) -> bool {
    let Some((last, body)) = block.insts.split_last() else {
        return false;
    };
    if !matches!(
        last.name(),
        "s_cbranch_vccz" | "s_cbranch_vccnz" | "s_cbranch_scc0" | "s_cbranch_scc1"
    ) {
        return false;
    }
    body.iter().all(|inst| {
        if inst
            .writes
            .iter()
            .any(|&(r, _)| r == Reg::S(EXEC) || r == Reg::S(EXEC + 1))
        {
            return false;
        }
        match &inst.fields {
            Fields::Sop { .. } => {
                let name = inst.name();
                !name.starts_with("s_setreg")
                    && !name.starts_with("s_set_")
                    && !name.starts_with("s_cbranch")
                    && !name.starts_with("s_swappc")
                    && !name.starts_with("s_setpc")
                    && !name.starts_with("s_call")
                    && !name.starts_with("s_rfe")
            }
            Fields::Valu(v) => {
                let name = inst.name();
                v.dpp.is_none()
                    && !name.starts_with("v_readlane")
                    && !name.starts_with("v_writelane")
                    && !name.starts_with("v_readfirstlane")
                    && !name.starts_with("v_permlane")
                    && !name.starts_with("v_movrel")
                    && !name.starts_with("v_swaprel")
            }
            _ => false,
        }
    })
}

/// Copies `block`'s non-branch instructions and drops those whose writes are
/// dead on entry to `target`.
fn prune_join_body(block: &Block, live_target: &RegSet) -> Vec<Inst> {
    let body = &block.insts[..block.insts.len() - 1];
    let mut live = *live_target;
    let mut keep = vec![false; body.len()];
    for (i, inst) in body.iter().enumerate().rev() {
        if inst.writes.iter().any(|&(r, _)| contains(&live, r)) {
            keep[i] = true;
            for &(r, _) in &inst.writes {
                remove(&mut live, r);
            }
            for &(r, _) in &inst.reads {
                insert(&mut live, r);
            }
        }
    }
    body.iter()
        .zip(keep)
        .filter(|&(_, k)| k)
        .map(|(inst, _)| inst.clone())
        .collect()
}

/// Known constant register values at the exit of `p`, gathered along its
/// unique single-predecessor chain (stopping if EXEC changes or the chain
/// splits).
fn incoming_constants(blocks: &[Block], p: usize, max_chain: usize) -> HashMap<Reg, u32> {
    let mut chain = vec![p];
    while chain.len() < max_chain {
        let cur = *chain.last().expect("non-empty");
        if blocks[cur].preds.len() != 1 {
            break;
        }
        let prev = blocks[cur].preds[0];
        if chain.contains(&prev) {
            break;
        }
        let ok = match blocks[prev].exit {
            Exit::Jump(t) => t == cur,
            Exit::Branch { taken, not_taken } if taken == cur || not_taken == cur => {
                blocks[prev].insts.last().is_some_and(|i| {
                    matches!(
                        i.name(),
                        "s_cbranch_scc0" | "s_cbranch_scc1" | "s_cbranch_vccz" | "s_cbranch_vccnz"
                    )
                })
            }
            _ => false,
        };
        if !ok {
            break;
        }
        chain.push(prev);
    }

    let mut consts = HashMap::new();
    for b in chain.into_iter().rev() {
        for inst in &blocks[b].insts {
            if inst
                .writes
                .iter()
                .any(|&(r, _)| r == Reg::S(EXEC) || r == Reg::S(EXEC + 1))
            {
                consts.retain(|r, _| !matches!(r, Reg::V(_)));
            }
            step_constants(inst, &mut consts);
        }
    }
    consts
}

/// Updates `consts` with any simple constant-producing move, invalidating all
/// other written registers.
fn step_constants(inst: &Inst, consts: &mut HashMap<Reg, u32>) {
    let name = inst.name();
    let known = match &inst.fields {
        Fields::Valu(v)
            if name == "v_mov_b32"
                && !v.clamp
                && v.omod == 0
                && v.neg == 0
                && v.abs == 0
                && v.sdwa.is_none()
                && v.dpp.is_none() =>
        {
            match (v.dst, v.src.first()) {
                (Some(Operand::Vector { reg, dwords: 1 }), Some(src)) => {
                    operand_const(src, consts).map(|val| (Reg::V(reg), val))
                }
                _ => None,
            }
        }
        Fields::Sop {
            dst: Some(Operand::Scalar { code, dwords: 1 }),
            src,
            imm,
        } if *code != NULL && *code != EXEC && *code != EXEC + 1 => match name {
            "s_mov_b32" => src
                .first()
                .and_then(|s| operand_const(s, consts))
                .map(|val| (Reg::S(*code), val)),
            "s_movk_i32" => imm.map(|k| (Reg::S(*code), i32::from(k as i16) as u32)),
            _ => None,
        },
        _ => None,
    };

    for &(r, _) in &inst.writes {
        consts.remove(&r);
    }
    if let Some((r, val)) = known {
        consts.insert(r, val);
    }
}

fn operand_const(op: &Operand, consts: &HashMap<Reg, u32>) -> Option<u32> {
    match *op {
        Operand::Literal(x) => Some(x),
        Operand::Const(c) => inline_const(c),
        Operand::Scalar { code: NULL, .. } => Some(0),
        Operand::Scalar { code, dwords: 1 } => consts.get(&Reg::S(code)).copied(),
        Operand::Vector { reg, dwords: 1 } => consts.get(&Reg::V(reg)).copied(),
        _ => None,
    }
}

fn inline_const(c: u8) -> Option<u32> {
    Some(match c {
        128..=192 => u32::from(c - 128),
        193..=208 => (192i32 - i32::from(c)) as u32,
        240 => 0.5f32.to_bits(),
        241 => (-0.5f32).to_bits(),
        242 => 1.0f32.to_bits(),
        243 => (-1.0f32).to_bits(),
        244 => 2.0f32.to_bits(),
        245 => (-2.0f32).to_bits(),
        246 => 4.0f32.to_bits(),
        247 => (-4.0f32).to_bits(),
        248 => 0x3E22_F983, // 1 / (2 * pi)
        _ => return None,
    })
}

/// Known bit values of a lane mask in active (`EXEC == 1`) and inactive
/// (`EXEC == 0`) lanes. Tracking both regions separately keeps wave-wide zero /
/// non-zero tests sound when `EXEC` is partial (e.g. `!EXEC` is not zero).
#[derive(Clone, Copy, PartialEq, Eq)]
struct MaskBits {
    active: Option<bool>,
    inactive: Option<bool>,
}

impl MaskBits {
    const UNKNOWN: Self = Self {
        active: None,
        inactive: None,
    };
    const ZERO: Self = Self {
        active: Some(false),
        inactive: Some(false),
    };
    const ALL_ONES: Self = Self {
        active: Some(true),
        inactive: Some(true),
    };
    const EXEC: Self = Self {
        active: Some(true),
        inactive: Some(false),
    };

    fn not(self) -> Self {
        Self {
            active: self.active.map(|b| !b),
            inactive: self.inactive.map(|b| !b),
        }
    }

    fn and(self, other: Self) -> Self {
        Self {
            active: bit_and(self.active, other.active),
            inactive: bit_and(self.inactive, other.inactive),
        }
    }

    fn or(self, other: Self) -> Self {
        Self {
            active: bit_or(self.active, other.active),
            inactive: bit_or(self.inactive, other.inactive),
        }
    }

    fn xor(self, other: Self) -> Self {
        Self {
            active: bit_xor(self.active, other.active),
            inactive: bit_xor(self.inactive, other.inactive),
        }
    }

    /// Whether the mask is definitely non-zero (`Some(true)`) or all-zero
    /// (`Some(false)`) whenever the wave has active lanes.
    fn is_nonzero(self) -> Option<bool> {
        match (self.active, self.inactive) {
            (Some(false), Some(false)) => Some(false),
            (Some(true), _) => Some(true),
            _ => None,
        }
    }
}

fn bit_and(a: Option<bool>, b: Option<bool>) -> Option<bool> {
    match (a, b) {
        (Some(false), _) | (_, Some(false)) => Some(false),
        (Some(true), Some(true)) => Some(true),
        _ => None,
    }
}

fn bit_or(a: Option<bool>, b: Option<bool>) -> Option<bool> {
    match (a, b) {
        (Some(true), _) | (_, Some(true)) => Some(true),
        (Some(false), Some(false)) => Some(false),
        _ => None,
    }
}

fn bit_xor(a: Option<bool>, b: Option<bool>) -> Option<bool> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x != y),
        _ => None,
    }
}

/// Evaluates `block`'s closing conditional branch given known constant
/// registers on entry; returns `Some(true)` if taken or `Some(false)` if not.
fn eval_join_branch(block: &Block, entry_consts: &HashMap<Reg, u32>, wave64: bool) -> Option<bool> {
    let (last, body) = block.insts.split_last()?;
    let width = if wave64 { "_b64" } else { "_b32" };
    let mut consts = entry_consts.clone();
    let mut masks: HashMap<u8, MaskBits> = HashMap::new();
    let mut scc_const: Option<bool> = None;
    let mut scc_mask: Option<MaskBits> = None;

    let mask_of =
        |op: &Operand, masks: &HashMap<u8, MaskBits>, consts: &HashMap<Reg, u32>| -> MaskBits {
            match *op {
                Operand::Scalar { code, .. } if code == EXEC => MaskBits::EXEC,
                Operand::Scalar { code: NULL, .. } => MaskBits::ZERO,
                Operand::Scalar { code, .. } => masks.get(&code).copied().unwrap_or_else(|| {
                    if !wave64 {
                        match consts.get(&Reg::S(code)) {
                            Some(&0) => MaskBits::ZERO,
                            Some(&u32::MAX) => MaskBits::ALL_ONES,
                            _ => MaskBits::UNKNOWN,
                        }
                    } else {
                        MaskBits::UNKNOWN
                    }
                }),
                Operand::Const(128) | Operand::Literal(0) => MaskBits::ZERO,
                Operand::Const(193) => MaskBits::ALL_ONES,
                Operand::Literal(u32::MAX) if !wave64 => MaskBits::ALL_ONES,
                _ => MaskBits::UNKNOWN,
            }
        };

    for inst in body {
        let name = inst.name();
        let mut new_mask: Option<(u8, MaskBits)> = None;
        let mut new_scc_const: Option<bool> = None;
        let mut new_scc_mask: Option<MaskBits> = None;

        match &inst.fields {
            Fields::Valu(v)
                if name.starts_with("v_cmp_")
                    && !v.clamp
                    && v.omod == 0
                    && v.sdwa.is_none()
                    && v.dpp.is_none()
                    && v.neg == 0
                    && v.abs == 0 =>
            {
                if let (Some(Operand::Scalar { code, .. }), [a, b]) = (v.dst, &v.src[..])
                    && code != NULL
                {
                    let active = operand_const(a, &consts)
                        .zip(operand_const(b, &consts))
                        .and_then(|(x, y)| eval_cmp(name, x, y));
                    // `v_cmp_*` always clears inactive lanes (`EXEC == 0`).
                    new_mask = Some((
                        code,
                        MaskBits {
                            active,
                            inactive: Some(false),
                        },
                    ));
                }
            }
            Fields::Sop { dst, src, .. } if name.starts_with("s_cmp_") => {
                if let [a, b] = &src[..]
                    && let (Some(x), Some(y)) =
                        (operand_const(a, &consts), operand_const(b, &consts))
                {
                    new_scc_const = eval_scmp(name, x, y);
                }
                let _ = dst;
            }
            Fields::Sop { dst, src, .. } => {
                if let Some(op) = name.strip_prefix("s_").and_then(|s| s.strip_suffix(width)) {
                    let res = match op {
                        "mov" => src
                            .first()
                            .map_or(MaskBits::UNKNOWN, |s| mask_of(s, &masks, &consts)),
                        "not" => src
                            .first()
                            .map_or(MaskBits::UNKNOWN, |s| mask_of(s, &masks, &consts).not()),
                        "and" => src.get(1).map_or(MaskBits::UNKNOWN, |b| {
                            mask_of(&src[0], &masks, &consts).and(mask_of(b, &masks, &consts))
                        }),
                        "or" => src.get(1).map_or(MaskBits::UNKNOWN, |b| {
                            mask_of(&src[0], &masks, &consts).or(mask_of(b, &masks, &consts))
                        }),
                        "xor" => src.get(1).map_or(MaskBits::UNKNOWN, |b| {
                            mask_of(&src[0], &masks, &consts).xor(mask_of(b, &masks, &consts))
                        }),
                        "andn2" => src.get(1).map_or(MaskBits::UNKNOWN, |b| {
                            mask_of(&src[0], &masks, &consts).and(mask_of(b, &masks, &consts).not())
                        }),
                        "orn2" => src.get(1).map_or(MaskBits::UNKNOWN, |b| {
                            mask_of(&src[0], &masks, &consts).or(mask_of(b, &masks, &consts).not())
                        }),
                        "andn1" => src.get(1).map_or(MaskBits::UNKNOWN, |b| {
                            mask_of(&src[0], &masks, &consts)
                                .not()
                                .and(mask_of(b, &masks, &consts))
                        }),
                        "orn1" => src.get(1).map_or(MaskBits::UNKNOWN, |b| {
                            mask_of(&src[0], &masks, &consts)
                                .not()
                                .or(mask_of(b, &masks, &consts))
                        }),
                        "nand" => src.get(1).map_or(MaskBits::UNKNOWN, |b| {
                            mask_of(&src[0], &masks, &consts)
                                .and(mask_of(b, &masks, &consts))
                                .not()
                        }),
                        "nor" => src.get(1).map_or(MaskBits::UNKNOWN, |b| {
                            mask_of(&src[0], &masks, &consts)
                                .or(mask_of(b, &masks, &consts))
                                .not()
                        }),
                        "xnor" => src.get(1).map_or(MaskBits::UNKNOWN, |b| {
                            mask_of(&src[0], &masks, &consts)
                                .xor(mask_of(b, &masks, &consts))
                                .not()
                        }),
                        "cselect" => {
                            if let [a, b] = &src[..] {
                                match scc_const {
                                    Some(true) => mask_of(a, &masks, &consts),
                                    Some(false) => mask_of(b, &masks, &consts),
                                    None => {
                                        let ma = mask_of(a, &masks, &consts);
                                        let mb = mask_of(b, &masks, &consts);
                                        MaskBits {
                                            active: if ma.active == mb.active {
                                                ma.active
                                            } else {
                                                None
                                            },
                                            inactive: if ma.inactive == mb.inactive {
                                                ma.inactive
                                            } else {
                                                None
                                            },
                                        }
                                    }
                                }
                            } else {
                                MaskBits::UNKNOWN
                            }
                        }
                        _ => MaskBits::UNKNOWN,
                    };
                    if let Some(Operand::Scalar { code, .. }) = *dst
                        && code != NULL
                    {
                        new_mask = Some((code, res));
                    }
                    if inst.writes.iter().any(|&(r, _)| r == Reg::Scc) {
                        new_scc_mask = Some(res);
                    }
                }
            }
            _ => {}
        }

        // Invalidate overwritten scalar mask pairs and SCC.
        for &(r, _) in &inst.writes {
            match r {
                Reg::S(c) => {
                    masks.remove(&c);
                    if wave64 && c > 0 {
                        masks.remove(&(c - 1));
                    }
                }
                Reg::Scc => {
                    scc_const = None;
                    scc_mask = None;
                }
                Reg::V(_) => {}
            }
        }
        step_constants(inst, &mut consts);
        if let Some((code, bits)) = new_mask
            && bits != MaskBits::UNKNOWN
        {
            masks.insert(code, bits);
        }
        if new_scc_const.is_some() {
            scc_const = new_scc_const;
        }
        if new_scc_mask.is_some() {
            scc_mask = new_scc_mask;
        }
    }

    match last.name() {
        "s_cbranch_vccnz" => masks.get(&VCC).copied().and_then(MaskBits::is_nonzero),
        "s_cbranch_vccz" => masks
            .get(&VCC)
            .copied()
            .and_then(MaskBits::is_nonzero)
            .map(|nz| !nz),
        "s_cbranch_scc1" => scc_const.or_else(|| scc_mask.and_then(MaskBits::is_nonzero)),
        "s_cbranch_scc0" => scc_const
            .or_else(|| scc_mask.and_then(MaskBits::is_nonzero))
            .map(|nz| !nz),
        _ => None,
    }
}

fn eval_cmp(name: &str, a: u32, b: u32) -> Option<bool> {
    let rest = name.strip_prefix("v_cmp_")?;
    Some(match rest {
        "eq_u32" | "eq_i32" => a == b,
        "ne_u32" | "lg_u32" | "ne_i32" | "lg_i32" => a != b,
        "lt_u32" => a < b,
        "le_u32" => a <= b,
        "gt_u32" => a > b,
        "ge_u32" => a >= b,
        "lt_i32" => (a as i32) < (b as i32),
        "le_i32" => (a as i32) <= (b as i32),
        "gt_i32" => (a as i32) > (b as i32),
        "ge_i32" => (a as i32) >= (b as i32),
        _ => {
            let (fa, fb) = (f32::from_bits(a), f32::from_bits(b));
            if fa.is_subnormal() || fb.is_subnormal() {
                return None;
            }
            match rest {
                "eq_f32" => fa == fb,
                "neq_f32" => fa != fb,
                "lg_f32" => fa != fb && !fa.is_nan() && !fb.is_nan(),
                "lt_f32" => fa < fb,
                "le_f32" => fa <= fb,
                "gt_f32" => fa > fb,
                "ge_f32" => fa >= fb,
                "o_f32" => !fa.is_nan() && !fb.is_nan(),
                "u_f32" => fa.is_nan() || fb.is_nan(),
                _ => return None,
            }
        }
    })
}

fn eval_scmp(name: &str, a: u32, b: u32) -> Option<bool> {
    let rest = name.strip_prefix("s_cmp_")?;
    Some(match rest {
        "eq_u32" | "eq_i32" => a == b,
        "lg_u32" | "lg_i32" => a != b,
        "lt_u32" => a < b,
        "le_u32" => a <= b,
        "gt_u32" => a > b,
        "ge_u32" => a >= b,
        "lt_i32" => (a as i32) < (b as i32),
        "le_i32" => (a as i32) <= (b as i32),
        "gt_i32" => (a as i32) > (b as i32),
        "ge_i32" => (a as i32) >= (b as i32),
        _ => return None,
    })
}

/// Registers live into and out of each block.
fn liveness(blocks: &[Block], order: &[usize]) -> (Vec<RegSet>, Vec<RegSet>) {
    let empty = [0; REGS.div_ceil(64)];
    let mut uses = vec![empty; blocks.len()];
    let mut defs = vec![empty; blocks.len()];
    for (b, block) in blocks.iter().enumerate() {
        for inst in &block.insts {
            for &(reg, _) in &inst.reads {
                if !contains(&defs[b], reg) {
                    insert(&mut uses[b], reg);
                }
            }
            for &(reg, _) in &inst.writes {
                insert(&mut defs[b], reg);
            }
        }
    }
    let mut live_in = vec![empty; blocks.len()];
    let mut live_out = vec![empty; blocks.len()];
    let mut changed = true;
    while changed {
        changed = false;
        for &b in order.iter().rev() {
            let mut out = empty;
            for s in blocks[b].exit.successors() {
                for (o, i) in out.iter_mut().zip(&live_in[s]) {
                    *o |= i;
                }
            }
            let mut inn = empty;
            for w in 0..inn.len() {
                inn[w] = uses[b][w] | out[w] & !defs[b][w];
            }
            if inn != live_in[b] || out != live_out[b] {
                (live_in[b], live_out[b]) = (inn, out);
                changed = true;
            }
        }
    }
    (live_in, live_out)
}

/// Phis are placed for every live register at every join; those whose
/// arguments are all one value (or the phi itself) are replaced by it.
fn remove_trivial_phis(blocks: &mut [Block], values: u32) {
    let mut replaced: Vec<Value> = (0..values).map(Value).collect();
    let find = |replaced: &[Value], mut v: Value| {
        while replaced[v.0 as usize] != v {
            v = replaced[v.0 as usize];
        }
        v
    };
    let mut changed = true;
    while changed {
        changed = false;
        for block in blocks.iter_mut() {
            block.phis.retain(|phi| {
                let mut only = None;
                for &a in &phi.args {
                    let a = find(&replaced, a);
                    if a == phi.value || Some(a) == only {
                        continue;
                    }
                    if only.is_some() {
                        return true;
                    }
                    only = Some(a);
                }
                let Some(v) = only else { return true };
                replaced[phi.value.0 as usize] = v;
                changed = true;
                false
            });
        }
    }
    for block in blocks {
        for phi in &mut block.phis {
            for a in &mut phi.args {
                *a = find(&replaced, *a);
            }
        }
        for inst in &mut block.insts {
            for (_, v) in &mut inst.reads {
                *v = find(&replaced, *v);
            }
        }
    }
}

/// Scheduling hints, performance counters and tracing: no effect on results.
const DROPPED: &[&str] = &[
    "s_nop",
    "s_waitcnt",
    "s_waitcnt_vscnt",
    "s_waitcnt_vmcnt",
    "s_waitcnt_expcnt",
    "s_waitcnt_lgkmcnt",
    "s_waitcnt_depctr",
    "s_wait_idle",
    "s_inst_prefetch",
    "s_clause",
    "s_setprio",
    "s_sleep",
    "s_ttracedata",
    "s_ttracedata_imm",
    "s_incperflevel",
    "s_decperflevel",
    "v_nop",
    "v_pipeflush",
    "v_clrexcp",
];

/// Registers read, registers written.
type Effects = (Vec<Reg>, Vec<Reg>);

/// Registers an instruction reads and writes, one per dword, or `None` if
/// the IR drops it.
fn effects(
    inst: &Instruction,
    fields: &Fields,
    name: &'static str,
    lanes: u8,
    vgprs: u16,
) -> Result<Option<Effects>, ShaderError> {
    if DROPPED.contains(&name) {
        return Ok(None);
    }
    let unsupported = || ShaderError::Unsupported {
        offset: inst.offset,
        what: name,
    };
    let mask = |code: u8| (0..lanes).map(move |i| Reg::S(code + i));
    let (mut r, mut w) = (Vec::new(), Vec::new());
    let regs = |op: &Operand, out: &mut Vec<Reg>| {
        add_regs(op, lanes, out).map_err(|_| ShaderError::Isa {
            offset: inst.offset,
            reason: "register range out of bounds",
        })
    };
    let vector = !matches!(
        fields,
        Fields::Sop { .. } | Fields::Sopp { .. } | Fields::Smem { .. }
    );
    match fields {
        Fields::Sop { dst, src, .. } => {
            if name.starts_with("s_movrel")
                || name.starts_with("s_subvector_loop")
                || matches!(name, "s_call_b64" | "s_rfe_b64")
            {
                return Err(unsupported());
            }
            for s in src {
                regs(s, &mut r)?;
            }
            if let Some(d) = dst {
                regs(d, &mut w)?;
                // These change (or may keep) the destination's old value.
                if name.starts_with("s_bitset")
                    || name.starts_with("s_cmov")
                    || matches!(name, "s_addk_i32" | "s_mulk_i32")
                {
                    regs(d, &mut r)?;
                }
            }
            if matches!(name, "s_addc_u32" | "s_subb_u32" | "s_cmovk_i32")
                || name.starts_with("s_cselect_")
                || name.starts_with("s_cmov_")
            {
                r.push(Reg::Scc);
            }
            if name.contains("saveexec") || name.contains("wrexec") {
                r.extend(mask(EXEC));
                w.extend(mask(EXEC));
            }
            if writes_scc(inst.encoding, name) {
                w.push(Reg::Scc);
            }
        }
        Fields::Sopp { .. } => match name {
            "s_cbranch_scc0" | "s_cbranch_scc1" => r.push(Reg::Scc),
            "s_cbranch_vccz" | "s_cbranch_vccnz" => r.extend(mask(VCC)),
            "s_cbranch_execz" | "s_cbranch_execnz" => r.extend(mask(EXEC)),
            "s_sendmsg" | "s_sendmsghalt" => r.push(Reg::S(M0)),
            "s_endpgm"
            | "s_endpgm_saved"
            | "s_endpgm_ordered_ps_done"
            | "s_code_end"
            | "s_branch"
            | "s_barrier"
            | "s_icache_inv"
            | "s_wakeup"
            | "s_round_mode"
            | "s_denorm_mode" => {}
            _ => return Err(unsupported()),
        },
        Fields::Smem {
            data,
            base,
            soffset,
            ..
        } => match name {
            "s_memtime" | "s_memrealtime" | "s_get_waveid_in_workgroup" => {
                regs(data.as_ref().ok_or_else(unsupported)?, &mut w)?;
            }
            "s_gl1_inv" | "s_dcache_inv" | "s_dcache_wb" => {}
            _ if name.contains("_load_") || name.contains("_store_") => {
                regs(base, &mut r)?;
                regs(soffset, &mut r)?;
                let data = data.as_ref().ok_or_else(unsupported)?;
                regs(
                    data,
                    if name.contains("_load_") {
                        &mut w
                    } else {
                        &mut r
                    },
                )?;
            }
            _ => return Err(unsupported()),
        },
        Fields::Valu(v) => {
            let (src_indexed, dst_indexed) = match name {
                "v_movrels_b32" => (true, false),
                "v_movreld_b32" => (false, true),
                "v_movrelsd_b32" | "v_movrelsd_2_b32" | "v_swaprel_b32" => (true, true),
                _ => (false, false),
            };
            // v[reg + M0] for any M0 that stays inside the program's VGPRs.
            let indexed = |op: &Operand, out: &mut Vec<Reg>| match *op {
                Operand::Vector { reg, .. } if u16::from(reg) < vgprs => {
                    out.extend((u16::from(reg)..vgprs).map(|r| Reg::V(r as u8)));
                    Ok(())
                }
                _ => Err(ShaderError::Isa {
                    offset: inst.offset,
                    reason: "relative VGPR base outside the program's VGPRs",
                }),
            };
            for (i, s) in v.src.iter().enumerate() {
                if i == 0 && src_indexed {
                    indexed(s, &mut r)?;
                } else {
                    regs(s, &mut r)?;
                }
            }
            // GFX10 compares into EXEC write nothing else.
            if name.starts_with("v_cmpx") {
                w.extend(mask(EXEC));
            } else if let Some(d) = &v.dst {
                if dst_indexed {
                    indexed(d, &mut w)?;
                } else {
                    regs(d, &mut w)?;
                }
            }
            if let Some(c) = &v.carry {
                regs(c, &mut w)?;
            }
            match name {
                "v_swap_b32" => {
                    regs(&v.src[0], &mut w)?;
                    if let Some(d) = &v.dst {
                        regs(d, &mut r)?;
                    }
                }
                "v_swaprel_b32" => {
                    indexed(&v.src[0], &mut w)?;
                    if let Some(d) = &v.dst {
                        indexed(d, &mut r)?;
                    }
                }
                _ => {}
            }
            if name.starts_with("v_interp_") || src_indexed || dst_indexed {
                r.push(Reg::S(M0));
            }
        }
        Fields::Vintrp { dst, src, .. } => {
            regs(src, &mut r)?;
            r.push(Reg::S(M0));
            regs(dst, &mut w)?;
        }
        Fields::Ds {
            dst,
            addr,
            data,
            gds,
            ..
        } => {
            for o in addr.iter().chain(data) {
                regs(o, &mut r)?;
            }
            if let Some(d) = dst {
                regs(d, &mut w)?;
            }
            if *gds
                || name.starts_with("ds_gws")
                || name.contains("addtid")
                || matches!(name, "ds_ordered_count" | "ds_append" | "ds_consume")
            {
                r.push(Reg::S(M0));
            }
        }
        Fields::Buffer {
            data,
            addr,
            rsrc,
            soffset,
            offen,
            idxen,
            lds,
            cache,
            ..
        } => {
            if *offen || *idxen {
                regs(addr, &mut r)?;
            }
            regs(rsrc, &mut r)?;
            regs(soffset, &mut r)?;
            if *lds {
                r.push(Reg::S(M0));
            }
            if let Some(d) = data {
                if name.contains("_load_") {
                    if !lds {
                        regs(d, &mut w)?;
                    }
                } else if name.contains("_store_") {
                    regs(d, &mut r)?;
                } else if name.contains("_atomic_") {
                    regs(d, &mut r)?;
                    if cache.glc {
                        regs(&atomic_result(d, name), &mut w)?;
                    }
                } else {
                    return Err(unsupported());
                }
            }
        }
        Fields::Image {
            data,
            addr,
            rsrc,
            sampler,
            cache,
            ..
        } => {
            for a in addr.iter().chain([rsrc]).chain(sampler) {
                regs(a, &mut r)?;
            }
            if name.contains("_store") {
                regs(data, &mut r)?;
            } else if name.contains("_atomic_") {
                regs(data, &mut r)?;
                if cache.glc {
                    regs(&atomic_result(data, name), &mut w)?;
                }
            } else {
                regs(data, &mut w)?;
            }
        }
        Fields::Flat {
            dst,
            addr,
            data,
            saddr,
            ..
        } => {
            for o in [addr, data, saddr].into_iter().flatten() {
                regs(o, &mut r)?;
            }
            if let Some(d) = dst {
                regs(d, &mut w)?;
            }
        }
        Fields::Exp { src, .. } => {
            for s in src.iter().flatten() {
                regs(s, &mut r)?;
            }
        }
    }
    if vector {
        r.extend(mask(EXEC));
        // Lanes disabled in EXEC keep the old value.
        r.extend(w.iter().filter(|w| matches!(w, Reg::V(_))));
    }
    let mut seen = Vec::with_capacity(r.len());
    r.retain(|reg| {
        let new = !seen.contains(reg);
        seen.push(*reg);
        new
    });
    Ok(Some((r, w)))
}

/// Scalar ALU opcodes that set SCC (carry, overflow, comparison, or "result
/// is not zero").
fn writes_scc(encoding: gfx10::Encoding, name: &str) -> bool {
    use gfx10::Encoding::*;
    match encoding {
        Sopc => true,
        Sopk => name.starts_with("s_cmpk_") || name == "s_addk_i32",
        _ => {
            let stem = name.split('_').nth(1).unwrap_or("");
            name.contains("saveexec")
                || name.contains("wrexec")
                || matches!(
                    stem,
                    "add"
                        | "sub"
                        | "addc"
                        | "subb"
                        | "min"
                        | "max"
                        | "and"
                        | "or"
                        | "xor"
                        | "andn2"
                        | "orn2"
                        | "nand"
                        | "nor"
                        | "xnor"
                        | "lshl"
                        | "lshr"
                        | "ashr"
                        | "bfe"
                        | "absdiff"
                        | "lshl1"
                        | "lshl2"
                        | "lshl3"
                        | "lshl4"
                        | "not"
                        | "wqm"
                        | "bcnt0"
                        | "bcnt1"
                        | "quadmask"
                        | "abs"
                )
        }
    }
}

/// What an atomic with `glc` returns: the old value (for compare-and-swap,
/// half the data operand).
fn atomic_result(data: &Operand, name: &str) -> Operand {
    match *data {
        Operand::Vector { reg, dwords } if name.contains("cmpswap") => Operand::Vector {
            reg,
            dwords: dwords / 2,
        },
        op => op,
    }
}

/// Appends the register dwords an operand names; reading `vccz`, `execz`,
/// `scc` or `lds_direct` reads the register behind it.
fn add_regs(op: &Operand, lanes: u8, out: &mut Vec<Reg>) -> Result<(), ()> {
    match *op {
        Operand::Scalar { code, .. } if code == gfx10::NULL => {}
        Operand::Scalar { code, dwords } => {
            let end = u16::from(code) + u16::from(dwords);
            if end > 128 {
                return Err(());
            }
            out.extend((code..end as u8).map(Reg::S));
        }
        Operand::Vector { reg, dwords } => {
            if u16::from(reg) + u16::from(dwords) > 256 {
                return Err(());
            }
            out.extend((0..dwords).map(|i| Reg::V(reg + i)));
        }
        Operand::Const(251) => out.extend((0..lanes).map(|i| Reg::S(VCC + i))),
        Operand::Const(252) => out.extend((0..lanes).map(|i| Reg::S(EXEC + i))),
        Operand::Const(253) => out.push(Reg::Scc),
        Operand::Const(254) => out.push(Reg::S(M0)),
        Operand::Const(_) | Operand::Literal(_) | Operand::Attr { .. } => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const S_ENDPGM: u32 = 0xBF81_0000;

    fn build_words(words: &[u32], wave64: bool) -> Program {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        build(&code, wave64, 256).unwrap()
    }

    fn input(p: &Program, reg: Reg) -> Value {
        p.inputs.iter().find(|i| i.0 == reg).expect("input").1
    }

    #[test]
    fn joins_get_phis_only_where_values_differ() {
        let code = [
            0xBF06_0100, // s_cmp_eq_u32 s0, s1
            0xBF85_0001, // s_cbranch_scc1 -> 12
            0xBE82_0380, // s_mov_b32 s2, 0
            0x7E00_0202, // 12: v_mov_b32 v0, s2
            0x7E02_0200, // v_mov_b32 v1, s0
            S_ENDPGM,
        ];
        for wave64 in [false, true] {
            let p = build_words(&code, wave64);
            let [b0, b1, b2] = &p.blocks[..] else {
                panic!("{:?}", p.blocks)
            };
            assert_eq!(
                b0.exit,
                Exit::Branch {
                    taken: 2,
                    not_taken: 1
                }
            );
            assert_eq!(b1.exit, Exit::Jump(2));
            assert_eq!(b2.preds, [0, 1]);
            assert_eq!(b0.insts[1].read(Reg::Scc), b0.insts[0].write(Reg::Scc));

            // s2 differs per path; s0, EXEC, v0 and v1 do not.
            let [phi] = &b2.phis[..] else {
                panic!("{:?}", b2.phis)
            };
            assert_eq!(phi.reg, Reg::S(2));
            assert_eq!(
                phi.args,
                [input(&p, Reg::S(2)), b1.insts[0].write(Reg::S(2)).unwrap()]
            );
            let [mov0, mov1, end] = &b2.insts[..] else {
                panic!()
            };
            assert_eq!(mov0.read(Reg::S(2)), Some(phi.value));
            assert_eq!(mov1.read(Reg::S(0)), Some(input(&p, Reg::S(0))));
            // The old v1 is read: lanes outside EXEC keep it.
            assert_eq!(mov1.read(Reg::V(1)), Some(input(&p, Reg::V(1))));
            assert_eq!(mov1.read(Reg::S(EXEC)), Some(input(&p, Reg::S(EXEC))));
            assert_eq!(mov1.read(Reg::S(EXEC + 1)).is_some(), wave64);
            assert_eq!((end.name(), b2.exit), ("s_endpgm", Exit::End));
        }
    }

    #[test]
    fn in_place_scalar_ops_read_their_destination() {
        let p = build_words(
            &[
                0xBE8D_1D94, // s_bitset1_b32 s13, 20
                0xB78E_0010, // s_addk_i32 s14, 0x10
                0xBE8F_0380, // s_mov_b32 s15, 0
                S_ENDPGM,
            ],
            false,
        );
        let [bitset, addk, mov, _] = &p.blocks[0].insts[..] else {
            panic!("{:?}", p.blocks[0].insts)
        };
        assert_eq!(bitset.read(Reg::S(13)), Some(input(&p, Reg::S(13))));
        assert_eq!(addk.read(Reg::S(14)), Some(input(&p, Reg::S(14))));
        assert_eq!(mov.read(Reg::S(15)), None);
    }

    #[test]
    fn loops_carry_values_through_phis() {
        let p = build_words(
            &[
                0xBE80_0380, // s_mov_b32 s0, 0
                0x8000_8100, // 4: s_add_u32 s0, s0, 1
                0xBF07_8400, // s_cmp_lg_u32 s0, 4
                0xBF85_FFFD, // s_cbranch_scc1 -> 4
                S_ENDPGM,
            ],
            true,
        );
        assert!(p.inputs.is_empty(), "{:?}", p.inputs);
        let [b0, b1, b2] = &p.blocks[..] else {
            panic!()
        };
        assert_eq!(b1.preds, [0, 1]);
        assert_eq!(
            b1.exit,
            Exit::Branch {
                taken: 1,
                not_taken: 2
            }
        );
        let [phi] = &b1.phis[..] else {
            panic!("{:?}", b1.phis)
        };
        let add = &b1.insts[0];
        assert_eq!(phi.reg, Reg::S(0));
        assert_eq!(
            phi.args,
            [
                b0.insts[0].write(Reg::S(0)).unwrap(),
                add.write(Reg::S(0)).unwrap()
            ]
        );
        assert_eq!(add.read(Reg::S(0)), Some(phi.value));
        assert_eq!(b1.insts[1].read(Reg::S(0)), add.write(Reg::S(0)));
        assert_eq!(b2.insts[0].name(), "s_endpgm");
    }

    #[test]
    fn drops_scheduling_and_follows_far_jumps() {
        let p = build_words(
            &[
                0xBF80_0000, // s_nop 0
                0xBE88_1F00, // s_getpc_b64 s[8:9]
                0x8008_FF08,
                20,          // s_add_u32 s8, s8, 20  -> 8 + 20 = 28
                0x8209_8009, // s_addc_u32 s9, s9, 0
                0xBE80_2008, // s_setpc_b64 s[8:9]
                0xFFFF_FFFF, // not code
                S_ENDPGM,
            ],
            false,
        );
        let names: Vec<_> = p.blocks[0].insts.iter().map(Inst::name).collect();
        assert_eq!(
            names,
            ["s_getpc_b64", "s_add_u32", "s_addc_u32", "s_setpc_b64"]
        );
        assert_eq!(p.blocks[0].exit, Exit::Jump(1));
        assert_eq!(p.blocks[1].offset, 28);
        // s_addc_u32 reads the carry s_add_u32 left in SCC.
        let i = &p.blocks[0].insts;
        assert_eq!(i[2].read(Reg::Scc), i[1].write(Reg::Scc));
    }

    #[test]
    fn reports_what_it_cannot_express() {
        let code: Vec<u8> = [0xBE80_2E01_u32, S_ENDPGM] // s_movrels_b32 s0, s1
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        assert_eq!(
            build(&code, false, 256),
            Err(ShaderError::Unsupported {
                offset: 0,
                what: "s_movrels_b32"
            })
        );
    }

    #[test]
    fn relative_vgpr_access_spans_up_to_the_vgpr_count() {
        let code: Vec<u8> = [
            0x7E04_8501_u32, // v_movreld_b32 v2, v1: v[2 + m0] = v1
            0x7E00_8704,     // v_movrels_b32 v0, v4: v0 = v[4 + m0]
            S_ENDPGM,
        ]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
        let p = build(&code, false, 6).unwrap();
        let [movreld, movrels, _] = &p.blocks[0].insts[..] else {
            panic!()
        };
        let vgprs = |regs: &[(Reg, Value)]| -> Vec<u8> {
            regs.iter()
                .filter_map(|r| match r.0 {
                    Reg::V(v) => Some(v),
                    _ => None,
                })
                .collect()
        };
        assert_eq!(vgprs(&movreld.writes), [2, 3, 4, 5]);
        // v1 is the source; v2-v5 are read for the lanes and indices not written.
        assert_eq!(vgprs(&movreld.reads), [1, 2, 3, 4, 5]);
        assert!(movreld.read(Reg::S(M0)).is_some());
        assert_eq!(vgprs(&movrels.writes), [0]);
        assert_eq!(vgprs(&movrels.reads), [4, 5, 0]);
        assert_eq!(movrels.read(Reg::V(4)), movreld.write(Reg::V(4)));
        assert!(build(&code, false, 4).is_err(), "v4 is past 4 VGPRs");
    }

    #[test]
    fn threads_correlated_flag_past_join_removing_mixed_phi() {
        // B0 sets sentinel v0 = 0 and may skip B1 (which defines s2 and sets
        // dynamic flag v0 = s5) straight to the join B2. B2 branches to B4
        // whenever v0 != -1 || s2 == 0, so the B0 -> B2 edge always jumps to
        // B4 and never reaches B3's use of s2. Threading B0 -> B4 eliminates
        // the mixed phi for s2.
        let p = build_words(
            &[
                0xBE82_3C6A, // 0:  s_and_saveexec_b32 s2, vcc_lo  (dead mask in s2)
                0x7E00_0280, // 4:  v_mov_b32 v0, 0                (sentinel flag)
                0xBF04_8000, // 8:  s_cmp_lt_i32 s0, 0
                0xBF85_0003, // 12: s_cbranch_scc1 -> 28 (B2)
                0xBE82_0381, // 16: s_mov_b32 s2, 1                (B1: payload)
                0x7E00_0205, // 20: v_mov_b32 v0, s5               (dynamic flag)
                0xBF82_0000, // 24: s_branch -> 28 (B2)
                0x7D84_00C1, // 28: v_cmp_eq_u32 vcc_lo, -1, v0    (B2: join)
                0x8903_C16A, // 32: s_xor_b32 s3, vcc_lo, -1
                0xBF06_8002, // 36: s_cmp_eq_u32 s2, 0
                0x8504_807E, // 40: s_cselect_b32 s4, exec_lo, 0
                0x8803_0403, // 44: s_or_b32 s3, s3, s4
                0x876A_037E, // 48: s_and_b32 vcc_lo, exec_lo, s3
                0xBF87_0001, // 52: s_cbranch_vccnz -> 60 (B4)
                0x7E02_0202, // 56: v_mov_b32 v1, s2               (B3: payload use)
                S_ENDPGM,    // 60: s_endpgm                       (B4)
            ],
            false,
        );
        // B0's taken edge now bypasses B2 directly to B4 (offset 60).
        let Exit::Branch { taken, not_taken } = p.blocks[0].exit else {
            panic!("{:?}", p.blocks[0].exit);
        };
        assert_eq!(p.blocks[taken].offset, 60);
        assert_eq!(p.blocks[not_taken].offset, 16);
        // B2 (offset 28) has only B1 as predecessor and no phis.
        let join = p.blocks.iter().find(|b| b.offset == 28).unwrap();
        assert_eq!(join.preds.len(), 1);
        assert!(join.phis.is_empty(), "{:?}", join.phis);
        // B3's read of s2 comes directly from B1's `s_mov_b32 s2, 1`.
        let b1 = &p.blocks[not_taken];
        let b3 = p.blocks.iter().find(|b| b.offset == 56).unwrap();
        assert_eq!(b3.insts[0].read(Reg::S(2)), b1.insts[0].write(Reg::S(2)));
    }

    #[test]
    fn threads_through_empty_nop_block_and_preserves_live_join_writes() {
        // Same correlated pattern, with an intermediate s_nop block (B_nop)
        // entered via `s_cbranch_vccnz` and a live `v_mov_b32 v1, 1.0` inside
        // the join block that must still run on the threaded path into B_exit.
        let p = build_words(
            &[
                0x7E00_0280, // 0:  v_mov_b32 v0, 0
                0x8AEA_007E, // 4:  s_andn2_b64 vcc, exec, s[0:1]
                0xBF87_0003, // 8:  s_cbranch_vccnz -> 24 (B_nop)
                0xBE82_0381, // 12: s_mov_b32 s2, 1                (B_load)
                0x7E00_0205, // 16: v_mov_b32 v0, s5               (dynamic flag)
                0xBF82_0001, // 20: s_branch -> 28 (B_join)
                0xBF80_0000, // 24: s_nop 0                        (B_nop)
                0x7D84_00C1, // 28: v_cmp_eq_u32 vcc, -1, v0       (B_join)
                0x7E02_02F2, // 32: v_mov_b32 v1, 1.0              (live at exit!)
                0x8AEA_6A7E, // 36: s_andn2_b64 vcc, exec, vcc
                0xBF87_0001, // 40: s_cbranch_vccnz -> 48 (B_exit)
                0x7E04_0202, // 44: v_mov_b32 v2, s2               (B_use)
                0x7E06_0301, // 48: v_mov_b32 v3, v1               (B_exit: reads v1)
                S_ENDPGM,    // 52: s_endpgm
            ],
            true,
        );
        // B_join (offset 28) keeps only B_load as predecessor; B_nop jumps
        // through a cloned block carrying only the live `v_mov_b32 v1, 1.0`.
        let nop = p.blocks.iter().find(|b| b.offset == 24).unwrap();
        let Exit::Jump(copy_idx) = nop.exit else {
            panic!("{:?}", nop.exit);
        };
        let copy = &p.blocks[copy_idx];
        let names: Vec<_> = copy.insts.iter().map(Inst::name).collect();
        assert_eq!(names, ["v_mov_b32"]);
        let Exit::Jump(exit_idx) = copy.exit else {
            panic!("{:?}", copy.exit);
        };
        assert_eq!(p.blocks[exit_idx].offset, 48);
        let orig_join = p
            .blocks
            .iter()
            .enumerate()
            .find(|&(i, b)| i != copy_idx && b.offset == 28)
            .unwrap()
            .1;
        assert_eq!(orig_join.preds.len(), 1);
        assert!(orig_join.phis.is_empty(), "{:?}", orig_join.phis);
    }

    #[test]
    fn does_not_thread_across_exec_write_in_predecessor() {
        // Writing EXEC after `v_mov_b32 v0, 0` invalidates the vector constant
        // (lanes reactivated by the EXEC write did not execute the move).
        let p = build_words(
            &[
                0x7E00_0280, // 0:  v_mov_b32 v0, 0
                0xBEFE_0305, // 4:  s_mov_b32 exec_lo, s5          (modifies EXEC)
                0xBF04_8000, // 8:  s_cmp_lt_i32 s0, 0
                0xBF85_0002, // 12: s_cbranch_scc1 -> 24 (B_join)
                0x7E00_0206, // 16: v_mov_b32 v0, s6               (dynamic flag)
                0xBF82_0000, // 20: s_branch -> 24 (B_join)
                0x7D84_00C1, // 24: v_cmp_eq_u32 vcc_lo, -1, v0
                0xBF86_0001, // 28: s_cbranch_vccz -> 36
                0x7E02_0280, // 32: v_mov_b32 v1, 0
                S_ENDPGM,    // 36: s_endpgm
            ],
            false,
        );
        let join = p.blocks.iter().find(|b| b.offset == 24).unwrap();
        assert_eq!(join.preds.len(), 2, "must not thread past EXEC write");
    }

    #[test]
    fn does_not_fold_inverted_exec_mask_without_masking() {
        // When v0 == -1, `v_cmp_eq_u32 vcc_lo, -1, v0` yields EXEC, so
        // `s_xor_b32 s3, vcc_lo, -1` yields `!EXEC`, which is non-zero if
        // any lane is inactive and zero if all lanes are active. It must NOT
        // be folded as zero at `s_cbranch_scc1`.
        let p = build_words(
            &[
                0x7E00_02C1, // 0:  v_mov_b32 v0, -1
                0xBF04_8000, // 4:  s_cmp_lt_i32 s0, 0
                0xBF85_0002, // 8:  s_cbranch_scc1 -> 20 (B_join)
                0x7E00_0206, // 12: v_mov_b32 v0, s6
                0xBF82_0000, // 16: s_branch -> 20 (B_join)
                0x7D84_00C1, // 20: v_cmp_eq_u32 vcc_lo, -1, v0
                0x8903_C16A, // 24: s_xor_b32 s3, vcc_lo, -1
                0xBF85_0001, // 28: s_cbranch_scc1 -> 36
                0x7E02_0280, // 32: v_mov_b32 v1, 0
                S_ENDPGM,    // 36: s_endpgm
            ],
            false,
        );
        let join = p.blocks.iter().find(|b| b.offset == 20).unwrap();
        assert_eq!(join.preds.len(), 2, "must not fold !EXEC to zero");
    }
}
