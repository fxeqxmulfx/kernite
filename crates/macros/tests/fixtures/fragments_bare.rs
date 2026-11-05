//! A fragment that is no group, `$i:ident` and `$t:tt`, arrives as the tokens it is, and is read as it is written: a
//! name for a kernel, a variable and a constant; an operator, which `$t:tt` can be; a group in brackets, which is one
//! token tree and has its brackets. A literal of one token is a group of one, and is opened. A fragment that no
//! operator is next to, or that holds no operator of its own, is opened too, and the kernel is as it is written without
//! a macro.

/// How far the kernel `k`, which gives back the pointer it is given moved on, moved it.
#[cfg(test)]
fn moved(k: unsafe extern "sysv64" fn(*const u8) -> *const u8) -> isize {
    let bytes = [0u8; 16];
    let p = bytes.as_ptr();
    // SAFETY: `k` only moves the pointer it is given, and reads nothing through it.
    (unsafe { k(p) } as isize).wrapping_sub(p as isize)
}

/// A kernel of a name the macro is given, with a variable of a name, an operator and a group in brackets.
macro_rules! named {
    ($module:ident, $kernel:ident, $variable:ident, $constant:ident, $operator:tt, $group:tt) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub const $constant: usize = 4;

            pub unsafe extern "sysv64" fn $kernel(p: *const u8) -> *const u8 {
                let $variable = $constant;
                add(p, $variable $operator 2);
                add(p, $group * 10);
                ret(p);
            }
        }
    };
}

named!(bare, step, scale, FOUR, *, (1 + 2));

#[test]
fn names_operators_and_groups_in_brackets_are_read_as_they_were() {
    // 4 * 2, and (1 + 2) * 10.
    assert_eq!(moved(bare::step), 8 + 30);
    assert_eq!(bare::FOUR, 4);
}

/// A kernel whose operand is a literal, which is a group of one token, or of two for a number with a sign.
macro_rules! literal {
    ($module:ident, $test:ident, $l:literal) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn times(p: *const u8) -> *const u8 {
                add(p, $l * 2);
                ret(p);
            }
            pub unsafe extern "sysv64" fn minus(p: *const u8) -> *const u8 {
                add(p, 10 - $l);
                ret(p);
            }
            pub unsafe extern "sysv64" fn alone(p: *const u8) -> *const u8 {
                add(p, $l);
                ret(p);
            }
        }

        #[test]
        fn $test() {
            assert_eq!(moved($module::times), $l * 2);
            assert_eq!(moved($module::minus), 10 - $l);
            assert_eq!(moved($module::alone), $l);
        }
    };
}

literal!(literal_three, a_literal_is_read_as_it_was, 3);
literal!(literal_minus_three, a_literal_with_a_sign_is_read_as_rust_reads_it, -3);

/// A kernel whose operands are fragments that no operator is next to, or that hold none of their own: opened, each
/// is the tokens it is.
macro_rules! closed {
    ($module:ident, $test:ident, $path:expr, $other:expr) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn path(p: *const u8) -> *const u8 {
                add(p, $path - 1);
                ret(p);
            }
            pub unsafe extern "sysv64" fn alone(p: *const u8) -> *const u8 {
                add(p, $other);
                ret(p);
            }
        }

        #[test]
        fn $test() {
            assert_eq!(moved($module::path), $path as isize - 1);
            assert_eq!(moved($module::alone), $other);
        }
    };
}

closed!(closed_path, a_path_and_a_sum_alone_are_read_as_they_were, u8::MAX, 1 + 2);

/// A kernel whose statement is a fragment of several tokens, a `let`, which has operators of its own and none next
/// to it: it is the statement it is.
macro_rules! statement {
    ($module:ident, $test:ident, $statement:stmt) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn k(p: *const u8) -> *const u8 {
                $statement;
                add(p, k);
                ret(p);
            }
        }

        #[test]
        fn $test() {
            assert_eq!(moved($module::k), 5);
        }
    };
}

statement!(statement_let, a_statement_of_several_tokens_is_read_as_it_was, let k = 2 + 3);

/// A GPU kernel whose instruction begins with a fragment of several tokens that hold no operator, `st.global`, and
/// an operator is next to it, the `.` that follows.
macro_rules! instruction {
    ($module:ident, $test:ident, $space:expr) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "ptx-kernel" fn k(p: *mut u32) {
                $space.u32([p], 1);
            }
        }

        #[test]
        fn $test() {
            assert!($module::PTX.contains("st.global.u32 [%p], 1;"), "{}", $module::PTX);
        }
    };
}

instruction!(instruction_store, an_instruction_begun_by_a_fragment_is_read_as_it_was, st.global);
