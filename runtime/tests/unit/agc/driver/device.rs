use super::*;

#[test]
fn concurrent_submits_wait_for_their_own_acceptance() {
    let (tasks, incoming) = mpsc::sync_channel(QUEUE_DEPTH);
    let service = Arc::new(DeviceService { tasks, thread: std::thread::current().id() });
    let (finished, completions) = mpsc::channel();
    let threads: Vec<_> = [0x1000, 0x2000].into_iter().map(|address| {
        let service = service.clone();
        let finished = finished.clone();
        std::thread::spawn(move || {
            finished.send((address, service.submit(0, address, 1))).unwrap();
        })
    }).collect();
    let first = incoming.recv_timeout(Duration::from_secs(5)).unwrap();
    let second = incoming.recv_timeout(Duration::from_secs(5)).unwrap();
    let Task::Submit { address: first_address, accepted: first_ack, .. } = first else { panic!("expected submission") };
    let Task::Submit { address: second_address, accepted: second_ack, .. } = second else { panic!("expected submission") };
    second_ack.send(()).unwrap();
    let (address, outcome) = completions.recv_timeout(Duration::from_secs(5)).unwrap();
    assert_eq!(address, second_address);
    assert!(outcome.is_ok());
    assert!(matches!(completions.recv_timeout(Duration::from_millis(50)), Err(RecvTimeoutError::Timeout)));
    drop(first_ack);
    let (address, outcome) = completions.recv_timeout(Duration::from_secs(5)).unwrap();
    assert_eq!(address, first_address);
    assert_eq!(outcome.unwrap_err().code, result::SHUTDOWN);
    for thread in threads { thread.join().unwrap(); }
}
