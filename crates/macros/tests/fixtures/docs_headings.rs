//! Kernels whose docs have headings. The macros take a line that is a heading of the word Safety, `# Safety` or
//! `## Safety`, for the docs' `# Safety`, and not those words in a sentence or in a code block. What they add to a
//! `# Safety` comes at its end, before the next heading of its level or higher. A paragraph of their own that is no
//! part of a `# Safety` comes at the end of the text before the docs' first heading. A `# Safety` of their own comes at
//! the end of the docs. None of them joins the text after it, a title with a line of `=` under it say, nor ends a block
//! quote or a list item that holds a heading: there it comes where the block begins, or after it.

/// x86 kernels.
#[kernite_macros::kernels]
pub mod cpu {
    /// A word of the module's memory that kernels name.
    pub static WORD: global<u64>;

    /// Names `WORD`; its docs say `# Safety` in a sentence, and have no such heading.
    pub unsafe extern "sysv64" fn named() {
        let n: gpr;
        mov(n, [WORD]);
        inc(n);
        mov([WORD], n);
    }

    /// Names `WORD`; its docs have `# Safety` in a code block alone, and no such heading.
    ///
    /// ```text
    /// # Safety
    /// ```
    pub unsafe extern "sysv64" fn fenced() {
        let n: gpr;
        mov(n, [WORD]);
        inc(n);
        mov([WORD], n);
    }

    /// A cooperative kernel whose docs say `# Safety` in a sentence, and have no such heading.
    #[cooperative]
    pub unsafe extern "sysv64" fn joined() {}

    /// Names `WORD` and needs `popcnt`, with a `# Safety` and an `# Examples` after it.
    ///
    /// # Safety
    ///
    /// The kernel reads nothing.
    ///
    /// # Examples
    ///
    /// Call it once.
    pub unsafe extern "sysv64" fn ordered() {
        let n: gpr;
        mov(n, [WORD]);
        popcnt(n, n);
        mov([WORD], n);
    }

    /// Names `WORD`, with a `## Safety` that has a `### Why` of its own, and an `## Examples` after it.
    ///
    /// ## Safety
    ///
    /// The kernel reads nothing.
    ///
    /// ### Why
    ///
    /// It only writes.
    ///
    /// ## Examples
    ///
    /// Call it once.
    pub unsafe extern "sysv64" fn nested() {
        let n: gpr;
        mov(n, [WORD]);
        inc(n);
        mov([WORD], n);
    }

    /// Names `WORD`, with a `Safety` underlined by `=`, and an `# Examples` after it.
    ///
    /// Safety
    /// ======
    ///
    /// The kernel reads nothing.
    ///
    /// # Examples
    ///
    /// Call it once.
    pub unsafe extern "sysv64" fn setext_one() {
        let n: gpr;
        mov(n, [WORD]);
        inc(n);
        mov([WORD], n);
    }

    /// Names `WORD`, with a `Safety` underlined by `-`, a `### Why` in it, and an `## Examples` after it.
    ///
    /// Safety
    /// ------
    ///
    /// The kernel reads nothing.
    ///
    /// ### Why
    ///
    /// It only writes.
    ///
    /// ## Examples
    ///
    /// Call it once.
    pub unsafe extern "sysv64" fn setext_two() {
        let n: gpr;
        mov(n, [WORD]);
        inc(n);
        mov([WORD], n);
    }

    #[doc = "Names `WORD`, with its docs in one `#[doc]`.\n\n# Safety\n\nThe kernel reads nothing.\n\n\
             # Examples\n\nCall it once."]
    pub unsafe extern "sysv64" fn one_doc() {
        let n: gpr;
        mov(n, [WORD]);
        inc(n);
        mov([WORD], n);
    }

    /// A cooperative kernel that names `WORD` and needs `popcnt`, with a `# Safety` and an `# Examples` after it.
    ///
    /// # Safety
    ///
    /// The kernel reads nothing.
    ///
    /// # Examples
    ///
    /// Run it on two threads.
    #[cooperative]
    pub unsafe extern "sysv64" fn ordered_together() {
        let n: gpr;
        mov(n, [WORD]);
        popcnt(n, n);
        mov([WORD], n);
    }

    /// Names `WORD`; its `# Safety` ends in code, and an `Examples` with a line of `=` under it follows.
    ///
    /// # Safety
    ///
    /// The kernel reads nothing.
    ///
    /// ```text
    /// Nothing else.
    /// ```
    /// Examples
    /// ========
    ///
    /// Call it once.
    pub unsafe extern "sysv64" fn titled_after_code() {
        let n: gpr;
        mov(n, [WORD]);
        inc(n);
        mov([WORD], n);
    }

    /// Names `WORD`; its `# Safety` ends in code, and a list that begins at 2, with an `# Examples` in it, follows.
    ///
    /// # Safety
    ///
    /// The kernel reads nothing.
    ///
    /// ```text
    /// Nothing else.
    /// ```
    /// 2. # Examples
    ///
    ///    Call it once.
    pub unsafe extern "sysv64" fn listed_after_code() {
        let n: gpr;
        mov(n, [WORD]);
        inc(n);
        mov([WORD], n);
    }

    /// A cooperative kernel that names `WORD`; its `# Safety` ends in code, and a title with `=` under it follows.
    ///
    /// # Safety
    ///
    /// The kernel reads nothing.
    ///
    /// ```text
    /// Nothing else.
    /// ```
    /// Examples
    /// ========
    ///
    /// Call it once.
    #[cooperative]
    pub unsafe extern "sysv64" fn titled_after_code_together() {
        let n: gpr;
        mov(n, [WORD]);
        inc(n);
        mov([WORD], n);
    }

    /// Names `WORD`, with two `# Safety` in one list item.
    ///
    /// - # Safety
    ///
    ///   # Safety
    ///
    ///   The kernel reads nothing.
    pub unsafe extern "sysv64" fn twice() {
        let n: gpr;
        mov(n, [WORD]);
        inc(n);
        mov([WORD], n);
    }

    /// A cooperative kernel whose docs end with an example.
    ///
    /// # Examples
    ///
    /// Run it on two threads.
    #[cooperative]
    pub unsafe extern "sysv64" fn gathered() {}

    /// # Examples
    ///
    /// A cooperative kernel whose docs begin with a heading.
    #[cooperative]
    pub unsafe extern "sysv64" fn headed() {}

    /// Examples
    /// ========
    ///
    /// A cooperative kernel whose docs begin with a title that has a line of `=` under it.
    #[cooperative]
    pub unsafe extern "sysv64" fn titled() {}

    /// A cooperative kernel whose first heading is a title with a line of `=` in a list item.
    ///
    /// - Item
    ///
    ///   Title
    ///   =====
    #[cooperative]
    pub unsafe extern "sysv64" fn listed_title() {}

    /// A cooperative kernel that names `WORD`, whose first heading is a `# Safety` in a list item.
    ///
    /// - Item
    ///
    ///   # Safety
    ///
    ///   The kernel reads nothing.
    #[cooperative]
    pub unsafe extern "sysv64" fn listed() {
        let n: gpr;
        mov(n, [WORD]);
        inc(n);
        mov([WORD], n);
    }

    /// A cooperative kernel whose first heading is a `# Safety` in a block quote.
    ///
    /// > Quoted.
    /// >
    /// > # Safety
    /// >
    /// > The kernel reads nothing.
    #[cooperative]
    pub unsafe extern "sysv64" fn quoted() {}

    /// A cooperative kernel whose docs have a line of `#` in a code block before their first heading.
    ///
    /// ```text
    /// # Not a heading
    /// ```
    ///
    /// # Examples
    ///
    /// Run it on two threads.
    #[cooperative]
    pub unsafe extern "sysv64" fn blocked() {}
}

/// GPU kernels.
#[kernite_macros::kernels]
pub mod gpu {
    /// A GPU kernel whose docs say `# Safety` in a sentence, and have no such heading.
    pub unsafe extern "ptx-kernel" fn idle() {}

    /// A cooperative GPU kernel whose docs end with an example.
    ///
    /// # Examples
    ///
    /// Launch it on one block.
    #[cooperative]
    pub unsafe extern "ptx-kernel" fn spread() {}

    /// Launching
    /// =========
    ///
    /// A cooperative GPU kernel whose docs begin with a title that has a line of `=` under it.
    #[cooperative]
    pub unsafe extern "ptx-kernel" fn titled() {}

    /// Launching
    /// =========
    ///
    /// A GPU kernel that is no `unsafe fn`, whose docs begin with a title that has a line of `=` under it.
    pub extern "ptx-kernel" fn titled_safe() {}

    /// A GPU kernel that is no `unsafe fn`, whose docs end with an example.
    ///
    /// # Examples
    ///
    /// Launch it on one block.
    pub extern "ptx-kernel" fn rest() {}
}
