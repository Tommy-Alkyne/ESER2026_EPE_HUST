//! 日志窗口的文本缓冲：追加式拼接 + 超过行数上限时从最前面整行裁掉。
//!
//! 训练一跑就是几百 epoch，日志行数没有上限的话会一直吃内存、也会让每帧渲染越来越
//! 慢。改成有界缓冲区，超过上限只丢最老的行——不影响“看最新进展”这个真实需求。

/// 有界的日志文本缓冲。`text()` 直接喂给 egui 的只读多行文本框。
pub struct LogBuffer {
    text: String,
    lines: usize,
    max_lines: usize,
}

impl LogBuffer {
    pub fn new(max_lines: usize) -> Self {
        Self {
            text: String::new(),
            lines: 0,
            max_lines: max_lines.max(1),
        }
    }

    pub fn clear(&mut self) {
        self.text.clear();
        self.lines = 0;
    }

    /// 追加一行（不含结尾换行符，本函数负责补上）。
    pub fn push_line(&mut self, line: &str) {
        self.text.push_str(line);
        self.text.push('\n');
        self.lines += 1;
        if self.lines > self.max_lines {
            self.trim_front();
        }
    }

    /// 丢弃最前面的若干整行，直到行数回到上限以内。
    fn trim_front(&mut self) {
        let overflow = self.lines - self.max_lines;
        let mut cut_at = None;
        let mut removed = 0;
        for (i, ch) in self.text.char_indices() {
            if ch == '\n' {
                removed += 1;
                if removed == overflow {
                    cut_at = Some(i + 1);
                    break;
                }
            }
        }
        if let Some(idx) = cut_at {
            self.text.drain(..idx);
            self.lines -= overflow;
        }
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn line_count(&self) -> usize {
        self.lines
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_line_appends_with_newline() {
        let mut buf = LogBuffer::new(100);
        buf.push_line("a");
        buf.push_line("b");
        assert_eq!(buf.text(), "a\nb\n");
        assert_eq!(buf.line_count(), 2);
    }

    #[test]
    fn clear_resets_everything() {
        let mut buf = LogBuffer::new(100);
        buf.push_line("a");
        buf.clear();
        assert_eq!(buf.text(), "");
        assert_eq!(buf.line_count(), 0);
    }

    #[test]
    fn trims_oldest_lines_when_over_cap() {
        let mut buf = LogBuffer::new(3);
        for i in 0..5 {
            buf.push_line(&i.to_string());
        }
        // 只保留最新的 3 行：2, 3, 4
        assert_eq!(buf.text(), "2\n3\n4\n");
        assert_eq!(buf.line_count(), 3);
    }

    #[test]
    fn does_not_trim_when_exactly_at_cap() {
        let mut buf = LogBuffer::new(3);
        buf.push_line("a");
        buf.push_line("b");
        buf.push_line("c");
        assert_eq!(buf.text(), "a\nb\nc\n");
        assert_eq!(buf.line_count(), 3);
    }

    #[test]
    fn cap_of_one_keeps_only_latest_line() {
        let mut buf = LogBuffer::new(1);
        buf.push_line("a");
        buf.push_line("b");
        assert_eq!(buf.text(), "b\n");
        assert_eq!(buf.line_count(), 1);
    }
}
