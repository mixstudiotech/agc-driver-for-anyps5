use shader::ir::{AddressForm, DescriptorTable, TableLink};

fn table() -> DescriptorTable {
    DescriptorTable { root: 0, root_form: AddressForm::Pointer, links: Vec::new(), offset: 0x2b8 }
}

#[test]
fn optional_tables_stop_at_unbound_roots_and_links() {
    let mut source = table();
    assert_eq!(source.bound_address(|_| 0, |_| panic!("unbound table must not be read")), None);
    source.root_form = AddressForm::Buffer;
    assert_eq!(source.bound_address(|sgpr| if sgpr == 1 { 0x0010_0000 } else { 0 }, |_| panic!("flags are not an address")), None);
    source.root_form = AddressForm::Pointer;
    source.links.push(TableLink { offset: 8, form: AddressForm::Pointer });
    let mut reads = Vec::new();
    assert_eq!(source.bound_address(|sgpr| if sgpr == 0 { 0x1000 } else { 0 }, |at| { reads.push(at); Some([0, 0]) }), None);
    assert_eq!(reads, [0x1008]);
}

#[test]
fn optional_tables_preserve_bound_addresses_and_scalar_alignment() {
    let mut source = table();
    source.links.push(TableLink { offset: 11, form: AddressForm::Buffer });
    source.offset = -5;
    let user = |sgpr| if sgpr == 0 { 0x1000 } else { 0 };
    let read = |at| { assert_eq!(at, 0x1008); Some([0x2000, 0x0010_0000]) };
    assert_eq!(source.bound_address(user, read), Some(0x1ff8));
    assert_eq!(source.address(user, read), Some(0x1ff8));
    source = table();
    assert_eq!(source.address(|_| 0, |_| None), Some(0x2b8));
}
