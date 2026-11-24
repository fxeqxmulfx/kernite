//! The shapes of docs that the macros take for having a `# Safety`, or for having none, as Clippy does: a heading of
//! the word Safety alone, `Safety` or `SAFETY`, in emphasis or not, in each form that Markdown has for one, as rustdoc
//! reads it, in a block quote or a list item or not; and not the words in a sentence, a title that says more, `#`s too
//! many or glued, a line of a code block, fenced, indented or left open, a line of a block of HTML, a line of `=` or
//! `-` that is no underline of the paragraph above it, nor a line of docs that all have six spaces, which read as code
//! once the macros add a line of their own, as rustdoc takes off less of their indent. A title of two lines that has
//! the word on one is none of theirs, though Clippy takes it for one: they write a `# Safety` of their own as well.
//!
//! Each rule of Markdown that says where a block begins or ends has a kernel too: spaces after the marks of a fence
//! close it, and text does not; an underline has up to three spaces before it; a bullet, a comment, a declaration and a
//! tag of a block end a paragraph, and a tag alone on its line and a number other than 1 do not; a tag has white space
//! before each attribute, and a value after `=`; a break is of marks and spaces; an item may be empty or begin in code,
//! its number has up to nine digits, and a line of spaces ends an empty one; and a tab goes to the next column of four.
//! The macros close a code block or a block of HTML that the docs leave open at the top level, and none in a block
//! quote or a list item, so a kernel whose docs end in one says where it is.
//!
//! White space other than a space has a kernel at each place it reads otherwise, as pulldown-cmark has it: a line
//! tabulation and a form feed are blank where a block begins, after a list marker or the marks of an underline, and in
//! and after a tag, but a line of them goes on a paragraph; only spaces may follow the marks that close a fence, or the
//! `#`s that close a heading; and a no-break space is text. rustdoc leaves a line of white space alone as it is, with
//! the space of its `///`, and counts a doc of a macro's text, which the macros take for a line of text with no space
//! before it.
//!
//! Each kernel names a static of its module, so that the macros write the contract of its memory: into the `# Safety`
//! that they take the docs for having, and in one of their own where they take the docs for having none. Clippy tells
//! of no kernel.

/// x86 kernels, one for each shape of docs.
#[kernite_macros::kernels]
pub mod cpu {
    /// A static that the kernels name.
    pub static NEAR: global<[u32; 4]>;

    /// Docs.
    ///
    /// # Safety
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn atx1() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// ## Safety
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn atx2() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// ###### Safety
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn atx6() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// # Safety ##
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn closed() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// #   Safety
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn spaced() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    #[doc = "Docs.\n\n#\tSafety\n\nReads nothing."]
    pub unsafe extern "sysv64" fn tab() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    ///    # Safety
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn indent3() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// # SAFETY
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn upper() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// # safety
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn lower() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// # *Safety*
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn emph() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// # **Safety**
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn strong() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// # `Safety`
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn code_span() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// # Safety notes
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn more() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// # Safety!
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn bang() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// # Safety#
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn hash_glued() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// #Safety
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn nospace() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// ####### Safety
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn seven() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// Safety
    /// ======
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn setext1() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// Safety
    /// ------
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn setext2() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// > # Safety
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn quote() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// - # Safety
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn list() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// ```text
    /// # Safety
    /// ```
    pub unsafe extern "sysv64" fn fenced() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// ~~~text
    /// # Safety
    /// ~~~
    pub unsafe extern "sysv64" fn tilde() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// ```
    /// # Safety
    /// let x = 1;
    /// ```
    pub unsafe extern "sysv64" fn rust_fence() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// ````text
    /// ```
    /// # Safety
    /// ````
    pub unsafe extern "sysv64" fn long_fence() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// ```text
    /// # Safety
    pub unsafe extern "sysv64" fn unclosed() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    ///     # Safety
    pub unsafe extern "sysv64" fn indented() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// ```text
    /// # Safety
    /// ```
    ///
    /// # Safety
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn after_fence() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs that name `# Safety` in a sentence.
    pub unsafe extern "sysv64" fn sentence() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// <h1>Safety</h1>
    pub unsafe extern "sysv64" fn html() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    #[doc = "Docs.\n\n# Safety\n\nReads nothing."]
    pub unsafe extern "sysv64" fn attr_newlines() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /** Docs.

    # Safety

    Reads nothing. */
    pub unsafe extern "sysv64" fn block() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// <div>
    /// # Safety
    /// </div>
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn html_block() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    ///      Docs.
    ///
    ///      # Safety
    ///
    ///      Reads nothing.
    pub unsafe extern "sysv64" fn shared_indent() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// <https://example.com> says more.
    /// # Safety
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn autolink() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// <b>Bold</b> text.
    /// # Safety
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn inline_tag() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// Text.
    /// <br/>
    /// # Safety
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn lone_tag() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// <!-- A note. -->
    /// # Safety
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn comment_line() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// <pre>a</pre>
    /// # Safety
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn pre_line() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// 1. # Safety
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn ordered() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// - Item.
    ///
    ///     # Safety
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn item_text() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// ```a`b
    /// # Safety
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn backtick_info() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// > ```text
    /// > code
    /// Text.
    ///
    /// # Safety
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn quote_fence() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// - Item:
    ///
    ///   ```text
    ///   code
    /// Text.
    ///
    /// # Safety
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn item_fence() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// <!-- A note.
    ///
    /// # Safety
    ///
    /// Reads nothing. -->
    pub unsafe extern "sysv64" fn comment_gap() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// <pre>
    /// a
    ///
    /// # Safety
    ///
    /// </pre>
    pub unsafe extern "sysv64" fn pre_gap() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// Text.
    /// <pre>
    /// # Safety
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn pre_open() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// <!-- A note.
    ///
    /// # Safety
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn comment_open() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// Safety
    /// > ---
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn quote_under() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// Safety
    /// - - -
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn break_under() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// > Item
    /// Safety
    /// ===
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn lazy_quote() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// - Item
    /// Safety
    /// ---
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn lazy_item() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// 1. ```text
    ///    # Safety
    ///    ```
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn ordered_fence() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    #[doc = "Docs.\n\n>\t```text\n> # Safety\n> ```\n\nReads nothing."]
    pub unsafe extern "sysv64" fn tab_fence() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// ```text
    /// > ```
    /// # Safety
    /// ```
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn fence_quote_line() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// > ```text
    /// > code
    pub unsafe extern "sysv64" fn quote_open() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// - Item:
    ///
    ///   ```text
    ///   code
    pub unsafe extern "sysv64" fn item_open() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    #[doc = "Docs.\n\n```text\nx\n```  \n\n# Safety\n\nReads nothing."]
    pub unsafe extern "sysv64" fn fence_spaces() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// ```text
    /// ``` not the end
    /// # Safety
    /// ```
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn fence_text() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    ///     code
    ///
    /// # Safety
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn after_indented() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// > Item
    /// lazy
    /// # Safety
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn after_lazy() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// Safety
    /// more
    /// ======
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn two_line_title() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// > Item
    /// Safety
    /// > ===
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn lazy_title() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// >> Safety
    /// > ===
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn nested_lazy() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    /// 2. # Safety
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn ordered_two() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    /// 1. # Safety
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn ordered_after_text() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// -     # Safety
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn item_code() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// Safety
    ///    ===
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn underline_spaces() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// Safety
    ///     ---
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn underline_four() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// <a >
    /// # Safety
    /// </a>
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn lone_tag_space() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// <a b='c'>
    /// # Safety
    /// </a>
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn lone_tag_quoted() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// <a href=x>
    /// # Safety
    /// </a>
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn lone_tag_value() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    #[doc = "Docs.\n\n-\n   \n   ```text\n   code"]
    pub unsafe extern "sysv64" fn spaces_after_empty_item() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// > Text
    /// -
    ///   ```text
    ///   code
    pub unsafe extern "sysv64" fn bullet_after_quote_text() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// * * *
    ///   ```text
    ///   code
    pub unsafe extern "sysv64" fn spaced_break_open() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// - -
    ///   ```text
    ///   code
    pub unsafe extern "sysv64" fn dashes_item_open() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    /// - Item
    ///   ```text
    ///   code
    pub unsafe extern "sysv64" fn bullet_after_text_open() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// -
    ///   ```text
    ///   code
    pub unsafe extern "sysv64" fn empty_item_open() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// -
    ///  ```text
    ///  code
    pub unsafe extern "sysv64" fn empty_item_short_open() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// -     code
    ///   ```text
    ///   code
    pub unsafe extern "sysv64" fn code_item_open() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// -     code
    ///  ```text
    ///  code
    pub unsafe extern "sysv64" fn code_item_short_open() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    /// <!--
    /// # Safety
    /// -->
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn comment_after_text() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    /// <!a
    /// # Safety
    /// >
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn declaration_after_text() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    /// <div>
    /// # Safety
    /// </div>
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn div_after_text() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    #[doc = "Docs.\n\n10. x\n\t```text\n\tcode"]
    pub unsafe extern "sysv64" fn tab_item_open() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    #[doc = "Docs.\n\n-\tSafety\n    ---\n\nReads nothing."]
    pub unsafe extern "sysv64" fn tab_item_title() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// ```text
    /// # Safety
    #[doc = " ```\u{a0}"]
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn fence_no_break() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// ```text
    /// # Safety
    #[doc = " ```\t"]
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn fence_tab() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// ```text
    /// # Safety
    #[doc = " ```\u{c}"]
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn fence_feed() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    #[doc = " - \u{a0}"]
    ///
    ///    ```text
    ///    code
    pub unsafe extern "sysv64" fn item_no_break() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// Safety
    #[doc = " ===\u{a0}"]
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn underline_no_break() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    #[doc = " # Safety #\u{a0}"]
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn closing_no_break() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    #[doc = " # Safety #\t"]
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn closing_tab() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    #[doc = " # Safety # "]
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn closing_spaces() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    #[doc = " #\u{c}Safety"]
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn hash_feed() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// # Safety
    ///
    /// Reads nothing.
    ///
    #[doc = " \u{c}# More"]
    pub unsafe extern "sysv64" fn feed_before_hash() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    #[doc = " <br>\u{a0}"]
    /// # Safety
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn lone_tag_no_break() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    #[doc = " <br> "]
    /// # Safety
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn lone_tag_spaces() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    #[doc = " <br\u{a0}b>"]
    /// # Safety
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn attribute_no_break() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    #[doc = " <br b=c\td='e'>"]
    /// # Safety
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn value_tab() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// <br b='c'd='e'>
    /// # Safety
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn values_glued() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// <br b=>
    /// # Safety
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn value_empty() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// <br b ='c'>
    /// # Safety
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn space_before_eq() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// Text.
    #[doc = " <hr\u{c}>"]
    /// # Safety
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn hr_feed() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// Text.
    /// <div
    /// class='a'>
    /// # Safety
    /// </div>
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn div_unended() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// Text.
    #[doc = " <pre\u{b}>"]
    /// # Safety
    /// </pre>
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn pre_tabulation() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    #[doc = " \u{c}"]
    /// <a>
    /// # Safety
    /// </a>
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn feed_before_tag() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// - ```text
    #[doc = " \u{c}"]
    ///   # Safety
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn item_feed_fence() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// Text.
    #[doc = " + \u{c}"]
    ///   ```text
    ///   code
    pub unsafe extern "sysv64" fn empty_item_feed() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// Text.
    /// +
    ///   ```text
    ///   code
    pub unsafe extern "sysv64" fn bare_plus() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// -
    ///   Text.
    ///
    ///     # Safety
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn empty_item_gap() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// -    x
    ///
    ///       # Safety
    ///
    ///       Reads nothing.
    pub unsafe extern "sysv64" fn item_four_spaces() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// 123456789. # Safety
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn nine_digits() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// 1234567890. # Safety
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn ten_digits() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    #[doc = " 1. \t# Safety"]
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn tab_after_marker() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    #[doc = " 10. \t# Safety"]
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn tab_after_ten() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// ``
    /// # Safety
    /// ``
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn two_ticks() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// - ```text
    #[doc = "  \u{c}"]
    ///   # Safety
    ///   ```
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn feed_in_fence() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// -   a
    ///
    #[doc = "    \u{c}"]
    ///     # Safety
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn feed_indent() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    #[doc = "\u{a0}"]
    ///    # Safety
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn no_break_line() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    #[doc = concat!("Text", ".")]
    ///
    ///    # Safety
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn macro_text() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// Text.
    #[allow(dead_code)]
    ///
    /// # Safety
    ///
    /// Reads nothing.
    pub unsafe extern "sysv64" fn attribute_between() {
        let p: gpr;
        lea(p, [NEAR]);
    }

    /// Docs.
    ///
    /// # Safety
    ///
    /// Reads nothing.
    #[doc = " \u{a0}"]
    pub unsafe extern "sysv64" fn no_break_last() {
        let p: gpr;
        lea(p, [NEAR]);
    }
}
