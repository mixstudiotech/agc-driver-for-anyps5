pub mod computed;
pub mod sources;

use crate::ir::opcodes::flag;
use crate::ir::{BasicBlock, Fields, InstructionFamily, Operand, RegFile, ShaderIr};
use crate::CompileError;

#[derive(Clone, Copy, Debug, Default)]
pub struct ControlFlowBuilder;

impl ControlFlowBuilder {
    pub fn new() -> Self {
        Self
    }

    pub fn build(&self, ir: &mut ShaderIr) -> Result<(), CompileError> {
        let count = ir.instructions.len();
        if count == 0 {
            return Err(CompileError::EmptyProgram);
        }
        let mut leader = vec![false; count];
        leader[0] = true;
        let entries: Vec<usize> = ir
            .flow
            .entries
            .iter()
            .filter_map(|&pc| ir.instruction_at(pc))
            .collect();
        for &index in &entries {
            leader[index] = true;
        }
        for (i, inst) in ir.instructions.iter().enumerate() {
            if let Some(target) = inst.branch_target {
                let index = ir
                    .instruction_at(target)
                    .ok_or(CompileError::InvalidBranch {
                        pc: inst.pc,
                        target,
                    })?;
                leader[index] = true;
            }
            if (inst.ends_block() || inst.has_flag(flag::CALL)) && i + 1 < count {
                leader[i + 1] = true;
            }
        }

        let mut blocks: Vec<BasicBlock> = Vec::new();
        for (i, &is_leader) in leader.iter().enumerate() {
            if is_leader {
                if let Some(last) = blocks.last_mut() {
                    last.end = i;
                }
                blocks.push(BasicBlock {
                    id: blocks.len(),
                    first: i,
                    end: count,
                    ..BasicBlock::default()
                });
            }
        }

        let block_of = |index: usize, blocks: &[BasicBlock]| {
            blocks
                .binary_search_by_key(&index, |b| b.first)
                .expect("branch targets are block leaders")
        };
        for id in 0..blocks.len() {
            let (first, end) = (blocks[id].first, blocks[id].end);
            let last = &ir.instructions[end - 1];
            blocks[id].start_pc = ir.instructions[first].pc;
            blocks[id].end_pc = last.pc + 4 * last.word_count() as u32;
            let mut successors = Vec::new();
            if let Some(target) = last.branch_target {
                let index = ir.instruction_at(target).expect("validated above");
                successors.push(block_of(index, &blocks));
            }
            if last.falls_through() && id + 1 < blocks.len() {
                successors.push(id + 1);
            }
            successors.dedup();
            blocks[id].successors = successors;
        }
        for id in 0..blocks.len() {
            for s in blocks[id].successors.clone() {
                if !blocks[s].predecessors.contains(&id) {
                    blocks[s].predecessors.push(id);
                }
            }
        }

        let mut stack = vec![0usize];
        if ir.flow.has_dynamic_jumps() {
            stack.extend(entries.iter().map(|&index| block_of(index, &blocks)));
        }
        while let Some(id) = stack.pop() {
            if blocks[id].reachable {
                continue;
            }
            blocks[id].reachable = true;
            stack.extend(blocks[id].successors.iter().copied());
        }

        ir.blocks = blocks;
        ir.metadata.user_data_tail_sgpr = tail_jump_sgpr(ir);
        Ok(())
    }
}

fn tail_jump_sgpr(ir: &ShaderIr) -> Option<u8> {
    let last_block = ir.blocks.iter().rev().find(|b| b.reachable)?;
    let jump = &ir.instructions[last_block.end - 1];
    if jump.family != InstructionFamily::Sop1 || !jump.has_flag(flag::JUMP) {
        return None;
    }
    let Fields::Sop1 {
        src0: Operand::Sgpr(base),
        ..
    } = jump.fields
    else {
        return None;
    };
    let written = ir
        .blocks
        .iter()
        .filter(|b| b.reachable)
        .flat_map(|b| b.instructions())
        .any(|i| {
            let mut writes = false;
            ir.instructions[i].each_def(true, |range| {
                writes |= range.contains(RegFile::Scalar, u16::from(base))
                    || range.contains(RegFile::Scalar, u16::from(base) + 1);
            });
            writes
        });
    (!written).then_some(base)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::Decoder;

    const S_ENDPGM: u32 = 0xbf81_0000;

    fn build(words: &[u32]) -> ShaderIr {
        let mut ir = Decoder::new().decode(words).expect("decode");
        ControlFlowBuilder::new().build(&mut ir).expect("cfg");
        ir
    }

    #[test]
    fn straight_line_code_is_one_block() {
        let ir = build(&[0xbe80_0380, S_ENDPGM]);
        assert_eq!(ir.blocks.len(), 1);
        assert_eq!(ir.blocks[0].start_pc, 0);
        assert_eq!(ir.blocks[0].end_pc, 8);
        assert!(ir.blocks[0].successors.is_empty());
        assert!(ir.blocks[0].reachable);
    }

    #[test]
    fn diamond_has_branch_and_fallthrough_edges() {
        let ir = build(&[
            0xbf06_8400,
            0xbf84_0002,
            0xbe81_0381,
            0xbf82_0001,
            0xbe81_0382,
            S_ENDPGM,
        ]);
        let starts: Vec<u32> = ir.blocks.iter().map(|b| b.start_pc).collect();
        assert_eq!(starts, vec![0, 8, 16, 20]);
        assert_eq!(ir.blocks[0].successors, vec![2, 1]);
        assert_eq!(ir.blocks[1].successors, vec![3]);
        assert_eq!(ir.blocks[2].successors, vec![3]);
        assert_eq!(ir.blocks[3].predecessors, vec![1, 2]);
        assert!(ir.blocks.iter().all(|b| b.reachable));
    }

    #[test]
    fn loop_back_edge_targets_its_own_block() {
        let ir = build(&[0xbe80_0380, 0x8000_8100, 0xbf06_8400, 0xbf84_fffd, S_ENDPGM]);
        assert_eq!(ir.blocks.len(), 3);
        assert_eq!(ir.blocks[1].start_pc, 4);
        assert_eq!(ir.blocks[1].successors, vec![1, 2]);
        assert_eq!(ir.blocks[1].predecessors, vec![0, 1]);
    }

    #[test]
    fn code_after_unconditional_branch_is_unreachable() {
        let ir = build(&[0xbf82_0001, 0xbe80_0380, S_ENDPGM]);
        assert_eq!(ir.blocks.len(), 3);
        assert!(!ir.blocks[1].reachable);
        assert!(ir.blocks[2].reachable);
    }

    #[test]
    fn detects_user_data_tail_jump() {
        let ir = build(&[0xbe80_0380, 0xbe80_2006]);
        assert_eq!(ir.metadata.user_data_tail_sgpr, Some(6));
        assert!(ir.blocks[0].successors.is_empty());
    }

    #[test]
    fn computed_jump_is_not_a_user_data_tail() {
        let ir = build(&[0xbe86_1f00, 0xbe80_2006]);
        assert_eq!(ir.metadata.user_data_tail_sgpr, None);
    }
}
