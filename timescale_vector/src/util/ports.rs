//! This module contains ports of Postgres static functions and #defines not in pgrx.
//! Following pgrx conventions, we keep function names as close to Postgres as possible.
//! Thus, we don't follow rust naming conventions.

use memoffset::*;
use pgrx::pg_sys::{Datum, ItemId, OffsetNumber, Pointer, TupleTableSlot};
use pgrx::{pg_sys, PgBox};

/// Given a valid Page pointer, return address of the "Special Pointer" (custom info at end of page)
///
/// # Safety
///
/// This function cannot determine if the `page` argument is really a non-null pointer to a [`Page`].
#[inline(always)]
#[allow(non_snake_case)]
pub unsafe fn PageGetSpecialPointer(page: pgrx::pg_sys::Page) -> Pointer {
    // PageValidateSpecialPointer(page);
    // return (char *) page + ((PageHeader) page)->pd_special;
    PageValidateSpecialPointer(page);
    let header = page.cast::<pgrx::pg_sys::PageHeaderData>();
    page.cast::<std::os::raw::c_char>()
        .add((*header).pd_special as usize)
}

#[allow(non_snake_case)]
pub unsafe fn PageValidateSpecialPointer(page: pgrx::pg_sys::Page) {
    //Assert(page);
    //Assert(((PageHeader) page)->pd_special <= BLCKSZ);
    //Assert(((PageHeader) page)->pd_special >= SizeOfPageHeaderData);
    assert!(!page.is_null());
    let header = page.cast::<pgrx::pg_sys::PageHeaderData>();
    assert!((*header).pd_special <= pgrx::pg_sys::BLCKSZ as u16);
    assert!((*header).pd_special >= SizeOfPageHeaderData as u16);
}

#[allow(non_upper_case_globals)]
const SizeOfPageHeaderData: usize = offset_of!(pgrx::pg_sys::PageHeaderData, pd_linp);

#[allow(non_snake_case)]
pub unsafe fn PageGetContents(page: pgrx::pg_sys::Page) -> *mut std::os::raw::c_char {
    //return (char *) page + MAXALIGN(SizeOfPageHeaderData);
    page.cast::<std::os::raw::c_char>()
        .add(pgrx::pg_sys::MAXALIGN(SizeOfPageHeaderData))
}

#[allow(non_snake_case)]
pub unsafe fn PageGetItem(page: pgrx::pg_sys::Page, item_id: ItemId) -> *mut std::os::raw::c_char {
    //Assert(page);
    //Assert(ItemIdHasStorage(itemId));

    //return (Item) (((char *) page) + ItemIdGetOffset(itemId));
    assert!(!page.is_null());
    assert!((*item_id).lp_len() != 0);

    page.cast::<std::os::raw::c_char>()
        .add((*item_id).lp_off() as _)
}

#[allow(non_snake_case)]
pub unsafe fn PageGetItemId(page: pgrx::pg_sys::Page, offset: OffsetNumber) -> ItemId {
    //return &((PageHeader) page)->pd_linp[offsetNumber - 1];
    let header = page.cast::<pgrx::pg_sys::PageHeaderData>();
    (*header).pd_linp.as_mut_ptr().add((offset - 1) as _)
}

#[allow(non_snake_case)]
pub unsafe fn PageGetMaxOffsetNumber(page: pgrx::pg_sys::Page) -> usize {
    /*
    PageHeader	pageheader = (PageHeader) page;

    if (pageheader->pd_lower <= SizeOfPageHeaderData)
        return 0;
    else
        return (pageheader->pd_lower - SizeOfPageHeaderData) / sizeof(ItemIdData);
     */

    let header = page.cast::<pgrx::pg_sys::PageHeaderData>();

    if (*header).pd_lower as usize <= SizeOfPageHeaderData {
        0
    } else {
        ((*header).pd_lower as usize - SizeOfPageHeaderData)
            / std::mem::size_of::<pgrx::pg_sys::ItemIdData>()
    }
}

pub unsafe fn slot_getattr(
    slot: &PgBox<TupleTableSlot>,
    attnum: pg_sys::AttrNumber,
) -> Option<Datum> {
    /*
    static inline Datum
    slot_getattr(TupleTableSlot *slot, int attnum,
                 bool *isnull)
    {
        Assert(attnum > 0);

        if (attnum > slot->tts_nvalid)
            slot_getsomeattrs(slot, attnum);

        *isnull = slot->tts_isnull[attnum - 1];

        return slot->tts_values[attnum - 1];
    }
    */
    assert!(attnum > 0);

    if attnum > slot.tts_nvalid {
        pg_sys::slot_getsomeattrs_int(slot.as_ptr(), attnum as _);
        /*panic!(
            "invalid attributes not yet handled: {} out of {}",
            attnum, slot.tts_nvalid
        );*/
    }

    let index = (attnum - 1) as usize;

    if *slot.tts_isnull.add(index) {
        return None;
    }
    return Some(*slot.tts_values.add(index));
}
