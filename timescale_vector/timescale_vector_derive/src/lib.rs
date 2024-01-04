use proc_macro::TokenStream;
use quote::{format_ident, quote};

#[proc_macro_derive(Readable)]
pub fn readable_macro_derive(input: TokenStream) -> TokenStream {
    // Construct a representation of Rust code as a syntax tree
    // that we can manipulate
    let ast = syn::parse(input).unwrap();

    // Build the trait implementation
    impl_readable_macro(&ast)
}

#[proc_macro_derive(Writeable)]
pub fn writeable_macro_derive(input: TokenStream) -> TokenStream {
    let ast = syn::parse(input).unwrap();
    impl_writeable_macro(&ast)
}

fn impl_readable_macro(ast: &syn::DeriveInput) -> TokenStream {
    let name = &ast.ident;
    let readable_name = format_ident!("Readable{}", name);
    let archived_name = format_ident!("Archived{}", name);
    let gen = quote! {
        pub struct #readable_name<'a> {
            _rb: ReadableBuffer<'a>,
        }

        impl<'a> #readable_name<'a> {
            pub fn get_archived_node(&self) -> & #archived_name {
                // checking the code here is expensive during build, so skip it.
                // TODO: should we check the data during queries?
                //rkyv::check_archived_root::<Node>(self._rb.get_data_slice()).unwrap()
                unsafe { rkyv::archived_root::<#name>(self._rb.get_data_slice()) }
            }
        }

        impl #name {
            pub unsafe fn read<'a>(index: &'a PgRelation, index_pointer: ItemPointer) -> #readable_name<'a> {
                let rb = index_pointer.read_bytes(index);
                #readable_name { _rb: rb }
            }
        }
    };
    gen.into()
}

fn impl_writeable_macro(ast: &syn::DeriveInput) -> TokenStream {
    let name = &ast.ident;
    let writeable_name = format_ident!("Writable{}", name);
    let archived_name = format_ident!("Archived{}", name);
    let gen = quote! {
        pub struct #writeable_name<'a> {
            wb: WritableBuffer<'a>,
        }

        impl #archived_name {
            pub fn with_data(data: &mut [u8]) -> Pin<&mut #archived_name> {
                let pinned_bytes = Pin::new(data);
                unsafe { rkyv::archived_root_mut::<#name>(pinned_bytes) }
            }
        }

        impl<'a> #writeable_name<'a> {
            pub fn get_archived_node(&self) -> Pin<&mut #archived_name> {
                #archived_name::with_data(self.wb.get_data_slice())
            }

            pub fn commit(self) {
                self.wb.commit()
            }
        }

        impl #name {
            pub unsafe fn modify(index: &PgRelation, index_pointer: ItemPointer) -> #writeable_name {
                let wb = index_pointer.modify_bytes(index);
                #writeable_name { wb: wb }
            }

            pub fn write(&self, tape: &mut Tape) -> ItemPointer {
                let bytes = rkyv::to_bytes::<_, 256>(self).unwrap();
                unsafe { tape.write(&bytes) }
            }
        }
    };
    gen.into()
}
