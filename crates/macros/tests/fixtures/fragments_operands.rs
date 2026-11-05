//! A macro's fragment is one operand where an x86 kernel reads it, as Rust reads it: `$n * 2` with `$n` as `1 + 2` is
//! 6, and not the 5 that `1 + 2 * 2` is. Each kernel is written by a macro and called for fragments, and what it
//! gives is what Rust gives for the same fragment, worked out beside it.
//!
//! The fragments hold each operator an integer has, `+ - * / % << >> & ^ |`, and a sign, `-3` and `!3`, and each a bool
//! has, `&& || == != < > <= >= & | ^`. Each stands before and after each operator of an integer or of a bool, and
//! before a method, `.min(2)`, and `as`; but none of a comparison has a comparison after it, which Rust refuses to
//! chain. The numbers are chosen so that a fragment read as the tokens it is made of gives another value:
//! `1000 / 60 / 3` is 5, and `1000 / (60 / 3)` is 50.

/// How far the kernel `k`, which gives back the pointer it is given moved on, moved it.
#[cfg(test)]
fn moved(k: unsafe extern "sysv64" fn(*const u8) -> *const u8) -> isize {
    let bytes = [0u8; 16];
    let p = bytes.as_ptr();
    // SAFETY: `k` only moves the pointer it is given, and reads nothing through it.
    (unsafe { k(p) } as isize).wrapping_sub(p as isize)
}

/// A module that a macro writes: `scaled!(sum, 1 + 2)` is `add(p, 1 + 2 * 2)` to a reader that opens `$n`.
macro_rules! scaled {
    ($module:ident, $n:expr) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn k(p: *const u8) -> *const u8 {
                add(p, $n * 2);
                ret(p);
            }
        }
    };
}

scaled!(plain, 3);
scaled!(sum, 1 + 2);

#[test]
fn a_fragment_is_one_operand() {
    assert_eq!(moved(plain::k), 6, "3 * 2");
    assert_eq!(moved(sum::k), 6, "(1 + 2) * 2");
}

/// A macro called with a fragment that another macro passes on: the other's fragment is `$n:expr` in it, or the
/// tokens that it took apart, which the first reads as `$n:expr` again.
macro_rules! passes_on {
    ($module:ident, $n:expr) => {
        scaled!($module, $n);
    };
}

macro_rules! passes_tokens_on {
    ($module:ident, $($n:tt)*) => {
        scaled!($module, $($n)*);
    };
}

passes_on!(passed, 1 + 2);
passes_tokens_on!(passed_as_tokens, 1 + 2);

#[test]
fn a_fragment_passed_to_another_macro_is_still_one_operand() {
    assert_eq!(moved(passed::k), 6, "(1 + 2) * 2, passed on whole");
    assert_eq!(moved(passed_as_tokens::k), 6, "(1 + 2) * 2, passed on as tokens");
}

/// Kernels, one for each `name: operand` that a list gives, which move the pointer on by the operand; and a test that
/// each moves it as far as Rust works the operand out. An operand holds a fragment of a macro, which passes it on
/// as an operand in turn.
macro_rules! moves {
    ($module:ident, $test:ident, $($name:ident: $operand:expr,)*) => {
        #[kernite_macros::kernels]
        pub mod $module {
            $(
                pub unsafe extern "sysv64" fn $name(p: *const u8) -> *const u8 {
                    add(p, $operand);
                    ret(p);
                }
            )*
        }

        #[test]
        fn $test() {
            let got = [$((stringify!($name), moved($module::$name))),*];
            let want = [$((stringify!($name), $operand as isize)),*];
            assert_eq!(got, want);
        }
    };
}

/// Kernels that put `$a` and `$b`, integers, next to each operator of an integer, and `.`, on either side. The
/// shifts are of `$b`, which is small and has no sign.
macro_rules! operands {
    ($module:ident, $test:ident, $a:expr, $b:expr) => {
        moves! { $module, $test,
            times: $a * 2,
            times_by: 2 * $a,
            squared: $a * $a,
            plus: 1000 + $a,
            plus_after: $a + 1000,
            minus: 1000 - $a,
            minus_after: $a - 1000,
            negated: 1000 + -$a,
            divides: 1000 / $a,
            divides_after: $a / 3,
            remainder: 1000 % $a,
            remainder_after: $a % 4,
            shifted: 1 << $b,
            shifted_after: $b << 1,
            shifted_right: 1000 >> $b,
            shifted_right_after: $b >> 1,
            masked: 12 & $b,
            masked_after: $b & 12,
            xored: 12 ^ $b,
            xored_after: $b ^ 12,
            ored: 12 | $b,
            ored_after: $b | 12,
            inverted: !$b,
            smaller: ($b < $a) as u8,
            smallest: $a.min(2),
            largest: $a.max(2),
        }
    };
}

operands!(of_numbers, numbers_are_read_as_rust_reads_them, 5, 3);
operands!(of_sums, sums_are_read_as_rust_reads_them, 2 + 4, 2 + 1);
operands!(of_differences, differences_are_read_as_rust_reads_them, 9 - 3, 5 - 2);
operands!(of_products, products_are_read_as_rust_reads_them, 7 * 3, 3 * 1);
operands!(of_quotients, quotients_are_read_as_rust_reads_them, 60 / 3, 9 / 3);
operands!(of_remainders, remainders_are_read_as_rust_reads_them, 20 % 7, 10 % 7);
operands!(of_left_shifts, left_shifts_are_read_as_rust_reads_them, 1 << 3, 1 << 2);
operands!(of_right_shifts, right_shifts_are_read_as_rust_reads_them, 64 >> 2, 16 >> 2);
operands!(of_ands, ands_are_read_as_rust_reads_them, 12 & 10, 7 & 6);
operands!(of_xors, xors_are_read_as_rust_reads_them, 5 ^ 3, 6 ^ 2);
operands!(of_ors, ors_are_read_as_rust_reads_them, 256 | 5, 2 | 1);
operands!(of_casts, casts_are_read_as_rust_reads_them, 300 as i64, 7 as i64);
operands!(of_negatives, negatives_are_read_as_rust_reads_them, -3, 3);
operands!(of_nots, nots_are_read_as_rust_reads_them, !3, 3);

/// Kernels that cast `$a`, which has no sign, to a type that cuts it: `256 | 5 as u8` is 261, `(256 | 5) as u8` is 5.
macro_rules! casts {
    ($module:ident, $test:ident, $a:expr) => {
        moves! { $module, $test,
            cast: $a as u8,
            cast_twice: $a as u8 as u16,
        }
    };
}

casts!(of_a_number_cast, a_number_is_cast_as_rust_casts_it, 5);
casts!(of_ors_cast, ors_are_cast_as_rust_casts_them, 256 | 5);
casts!(of_shifts_cast, shifts_are_cast_as_rust_casts_them, 1 << 8);

/// Kernels whose operands put `$a`, which may be negative, under a `-`, and next to one.
macro_rules! signs {
    ($module:ident, $test:ident, $a:expr) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn under(p: *const u8) -> *const u8 {
                add(p, -$a);
                ret(p);
            }
            pub unsafe extern "sysv64" fn minus(p: *const u8) -> *const u8 {
                add(p, 10 - $a);
                ret(p);
            }
            pub unsafe extern "sysv64" fn tight(p: *const u8) -> *const u8 {
                add(p, 10-$a);
                ret(p);
            }
            pub unsafe extern "sysv64" fn minus_under(p: *const u8) -> *const u8 {
                add(p, 10 - -$a);
                ret(p);
            }
            pub unsafe extern "sysv64" fn times(p: *const u8) -> *const u8 {
                add(p, $a * 2);
                ret(p);
            }
        }

        #[test]
        fn $test() {
            let got = [
                moved($module::under),
                moved($module::minus),
                moved($module::tight),
                moved($module::minus_under),
                moved($module::times),
            ];
            let want: [isize; 5] = [-$a, 10 - $a, 10-$a, 10 - -$a, $a * 2];
            assert_eq!(got, want);
        }
    };
}

signs!(of_three, three_is_read_as_rust_reads_it, 3);
signs!(of_minus_three, minus_three_is_read_as_rust_reads_it, -3);
signs!(of_minus_one_minus_two, minus_one_minus_two_is_read_as_rust_reads_it, -1 - 2);

/// Kernels that choose by `$c`, which is a bool known now, next to each operator of a bool. The constants are the ones
/// for which a `$c` read as the tokens it is made of gives another value, as `false && (true || true)` is false and
/// `(false && true) || true` is true.
macro_rules! choices {
    ($module:ident, $test:ident, $c:expr) => {
        moves! { $module, $test,
            equal: (false == $c) as u8,
            unequal: (true != $c) as u8,
            both: (false && $c) as u8,
            both_after: ($c && false) as u8,
            masked: (false & $c) as u8,
            masked_after: ($c & false) as u8,
            xored: (true ^ $c) as u8,
            xored_after: ($c ^ true) as u8,
            ored: (true | $c) as u8,
            ored_after: ($c | true) as u8,
            inverted: (!$c) as u8,
        }
    };
}

/// Kernels that choose by `$c` next to `==` and `!=` after it. Rust refuses `$c == false` for a `$c` that is a
/// comparison, `1 == 2`, which comparisons that follow one cannot be chained to, so the fragments of comparisons have
/// none of these.
macro_rules! compared {
    ($module:ident, $test:ident, $c:expr) => {
        moves! { $module, $test,
            equal_after: ($c == false) as u8,
            unequal_after: ($c != true) as u8,
        }
    };
}

choices!(of_true, true_is_read_as_rust_reads_it, true);
choices!(of_logical_ands, logical_ands_are_read_as_rust_reads_them, false && false);
choices!(of_logical_ors, logical_ors_are_read_as_rust_reads_them, true || true);
choices!(of_equals, equals_are_read_as_rust_reads_them, 1 == 2);
choices!(of_unequals, unequals_are_read_as_rust_reads_them, 1 != 2);
choices!(of_lesses, lesses_are_read_as_rust_reads_them, 1 < 2);
choices!(of_greaters, greaters_are_read_as_rust_reads_them, 2 > 3);
choices!(of_at_mosts, at_mosts_are_read_as_rust_reads_them, 1 <= 2);
choices!(of_at_leasts, at_leasts_are_read_as_rust_reads_them, 2 >= 3);
choices!(of_bit_ands, bit_ands_are_read_as_rust_reads_them, true & false);
choices!(of_bit_xors, bit_xors_are_read_as_rust_reads_them, true ^ true);
choices!(of_bit_ors, bit_ors_are_read_as_rust_reads_them, false | true);

compared!(true_is_compared, true_is_compared_as_rust_compares_it, true);
compared!(logical_ands_are_compared, logical_ands_are_compared_as_rust_compares_them, false && false);
compared!(logical_ors_are_compared, logical_ors_are_compared_as_rust_compares_them, true || true);
compared!(bit_ands_are_compared, bit_ands_are_compared_as_rust_compares_them, true & false);
compared!(bit_xors_are_compared, bit_xors_are_compared_as_rust_compares_them, true ^ true);
compared!(bit_ors_are_compared, bit_ors_are_compared_as_rust_compares_them, false | true);

/// Kernels that loop over a range, or take its length, put `$a` or `$r` first.
macro_rules! methods {
    ($module:ident, $test:ident, $a:expr, $r:expr) => {
        #[kernite_macros::kernels]
        pub mod $module {
            pub unsafe extern "sysv64" fn how_many(p: *const u8) -> *const u8 {
                add(p, $r.len());
                ret(p);
            }
            pub unsafe extern "sysv64" fn backwards(p: *const u8) -> *const u8 {
                for k in $r.rev() {
                    add(p, k);
                    add(p, 1000);
                }
                ret(p);
            }
            pub unsafe extern "sysv64" fn turns(p: *const u8) -> *const u8 {
                for k in 0..$a * 2 {
                    add(p, k);
                }
                ret(p);
            }
        }

        #[test]
        fn $test() {
            let got = [moved($module::how_many), moved($module::backwards), moved($module::turns)];
            let want = [
                $r.len() as isize,
                $r.rev().map(|k| k + 1000).sum::<isize>(),
                (0..$a * 2).sum::<isize>(),
            ];
            assert_eq!(got, want);
        }
    };
}

methods!(of_a_number, a_number_has_methods_as_rust_has_them, 5, 1..4);
methods!(of_a_sum, a_sum_has_methods_as_rust_has_them, 5 + 3, 1 + 1..2 + 3);
