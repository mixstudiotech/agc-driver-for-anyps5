use super::*;
use crate::agc::header;
use crate::gpu::agc::side_table::{Branch, Chunk, SideTable};

/// Command memory: the stub's 4 dwords of NOP and its tail CHAIN at dwords 0..8, a child at
/// dwords 32..40.
struct Commands {
    words: Vec<u32>,
}

impl Commands {
    fn new() -> Self {
        let mut words = vec![0u32; 64];
        nop(&mut words);
        Self { words }
    }

    fn at(&self, dword: usize) -> u64 {
        self.words.as_ptr() as u64 + dword as u64 * 4
    }

    fn chain(&mut self, dword: usize, target: u64, dwords: u32) {
        self.words[dword] = header(pkt3::INDIRECT_BUFFER, 4, 0).unwrap();
        self.words[dword + 1] = target as u32;
        self.words[dword + 2] = (target >> 32) as u32;
        self.words[dword + 3] = dwords | IB_CHAIN;
    }
}

/// A recording of `[from, to)` in the index, with a CHAIN note at `jump`.
fn record(sender: &std::sync::mpsc::Sender<Arc<Chunk>>, from: u64, to: u64, jump: Option<u64>) -> Arc<SideTable> {
    let table = Arc::new(SideTable::default());
    if let Some(at) = jump {
        table.branches.push(Branch { at, kind: BranchKind::Chain });
    }
    let chunk = Arc::new(Chunk::new(from, to, table.clone()));
    chunk.extend(to);
    sender.send(chunk).unwrap();
    table
}

fn ranges(stretches: &[Stretch]) -> Vec<(u64, u64)> {
    stretches.iter().map(|stretch| (stretch.start, stretch.end)).collect()
}

#[test]
fn a_tail_chain_is_followed_into_the_commands_it_continues_in() {
    let mut commands = Commands::new();
    let child = commands.at(32);
    commands.chain(4, child, 8);
    let (mut index, sender) = ChunkIndex::new();
    let stub = record(&sender, commands.at(0), commands.at(4), None);
    let jump = record(&sender, commands.at(4), commands.at(8), Some(commands.at(4)));
    let leaf = record(&sender, commands.at(32), commands.at(40), None);
    let stretches = follow(&mut index, commands.at(0), 8).unwrap();
    // The jump does not run: the stub's piece ends before it.
    assert_eq!(ranges(&stretches), vec![(commands.at(0), commands.at(4)), (child, commands.at(40))]);
    assert!(Arc::ptr_eq(&stretches[0].owners[0].table, &stub));
    assert_eq!(stretches[0].owners.len(), 1);
    assert!(Arc::ptr_eq(&stretches[1].owners[0].table, &leaf));
    drop(jump);
}

#[test]
fn a_cleared_or_predicated_chain_is_not_followed() {
    let mut commands = Commands::new();
    let child = commands.at(32);
    commands.chain(4, child, 8);
    let (mut index, sender) = ChunkIndex::new();
    record(&sender, commands.at(0), commands.at(8), Some(commands.at(4)));
    record(&sender, commands.at(32), commands.at(40), None);
    // Predicated: it runs on the GPU when the predicate holds, with the whole range.
    commands.words[4] |= 1;
    assert_eq!(ranges(&follow(&mut index, commands.at(0), 8).unwrap()), vec![(commands.at(0), commands.at(8))]);
    // Cleared to a NOP (`sceAgcSetNop`): nothing to follow.
    nop(&mut commands.words[4..8]);
    assert_eq!(ranges(&follow(&mut index, commands.at(0), 8).unwrap()), vec![(commands.at(0), commands.at(8))]);
}

#[test]
fn a_chain_back_into_itself_is_an_error() {
    let mut commands = Commands::new();
    let stub = commands.at(0);
    commands.chain(4, stub, 8);
    let (mut index, sender) = ChunkIndex::new();
    record(&sender, commands.at(0), commands.at(8), Some(commands.at(4)));
    assert!(follow(&mut index, stub, 8).is_err());
}

#[test]
fn commands_no_recording_wrote_are_an_error() {
    let commands = Commands::new();
    let (mut index, _sender) = ChunkIndex::new();
    assert!(follow(&mut index, commands.at(0), 8).is_err());
}
