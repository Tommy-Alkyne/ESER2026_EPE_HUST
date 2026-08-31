//! 中文字体装载。
//!
//! **egui 内置的字体只有拉丁字符**（Ubuntu-Light / Hack / NotoEmoji），而本界面的
//! 标签全是中文——不装 CJK 字体的话每个汉字都渲染成「豆腐块」□，界面等于不可读。
//! 窗口标题不受影响，那是操作系统画的，所以光看标题正常会误以为没问题。
//!
//! 这个坑是实测撞出来的：GUI 交付时只验到「进程存活、无报错」，
//! 而字体缺失既不报错也不 panic，只是把字画成方块。
//!
//! # 为什么优先选 .ttf 而不是 .ttc
//!
//! `.ttc` 是字体集合，要额外指定 face 索引；`.ttf` 是单一字体，直接读进去就能用。
//! 微软雅黑（msyh）是 ttc，黑体（simhei）与等线（Deng）是 ttf，所以后两者排前面。

use eframe::egui;

use std::path::Path;

/// 候选字体，按「优先级」排序。前两个是纯 ttf，最后两个是 ttc 兜底。
const CANDIDATES: &[(&str, u32)] = &[
    (r"C:\Windows\Fonts\simhei.ttf", 0), // 黑体
    (r"C:\Windows\Fonts\Deng.ttf", 0),   // 等线
    (r"C:\Windows\Fonts\simkai.ttf", 0), // 楷体
    (r"C:\Windows\Fonts\msyh.ttc", 0),   // 微软雅黑
    (r"C:\Windows\Fonts\simsun.ttc", 0), // 宋体
    // 非 Windows 的常见位置，方便将来换平台
    ("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", 0),
    ("/System/Library/Fonts/PingFang.ttc", 0),
];

/// 装载结果，供上层在界面上提示用户。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FontStatus {
    Loaded(String),
    /// 一个都没找到。**必须让用户看见**，而不是任由界面变成一片方块
    Missing,
}

impl FontStatus {
    pub fn is_loaded(&self) -> bool {
        matches!(self, Self::Loaded(_))
    }
}

/// 把系统里的中文字体插到 egui 字体表最前面。
pub fn install(ctx: &egui::Context) -> FontStatus {
    for (path, index) in CANDIDATES {
        let Ok(bytes) = std::fs::read(path) else {
            continue;
        };
        let mut fonts = egui::FontDefinitions::default();
        let mut data = egui::FontData::from_owned(bytes);
        data.index = *index;
        fonts.font_data.insert("cjk".to_owned(), data.into());

        // 插到最前面：等宽与比例字体都要，否则日志区（Monospace）仍是方块
        for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
            fonts
                .families
                .entry(family)
                .or_default()
                .insert(0, "cjk".to_owned());
        }
        ctx.set_fonts(fonts);
        return FontStatus::Loaded(file_name(path));
    }
    eprintln!(
        "[fonts] 系统里没找到可用的中文字体，界面上的汉字会显示成方块。\n\
         已尝试：{}",
        CANDIDATES
            .iter()
            .map(|(p, _)| *p)
            .collect::<Vec<_>>()
            .join("、")
    );
    FontStatus::Missing
}

fn file_name(path: &str) -> String {
    Path::new(path)
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidates_are_plain_ttf_first() {
        // .ttc 是字体集合，要额外指定 face 索引，出问题时更难查；纯 .ttf 排前面
        let first_ttc = CANDIDATES.iter().position(|(p, _)| p.ends_with(".ttc"));
        let last_ttf = CANDIDATES.iter().rposition(|(p, _)| p.ends_with(".ttf"));
        if let (Some(ttc), Some(ttf)) = (first_ttc, last_ttf) {
            assert!(ttf < ttc, "纯 .ttf 的候选应当排在 .ttc 之前");
        }
    }

    #[test]
    fn at_least_one_candidate_exists_on_this_machine() {
        // 本机跑测试时如果一个候选都没有，界面必然是一片方块——与其等用户发现，
        // 不如在这里就红。换了平台要往 CANDIDATES 里补路径
        let found: Vec<&str> = CANDIDATES
            .iter()
            .map(|(p, _)| *p)
            .filter(|p| std::path::Path::new(p).exists())
            .collect();
        assert!(
            !found.is_empty(),
            "本机没有任何候选中文字体，界面会显示成方块。已尝试：{:?}",
            CANDIDATES.iter().map(|(p, _)| *p).collect::<Vec<_>>()
        );
    }

    #[test]
    fn file_name_strips_the_directory() {
        assert_eq!(file_name(r"C:\Windows\Fonts\simhei.ttf"), "simhei.ttf");
        assert_eq!(file_name("/usr/share/fonts/x.ttc"), "x.ttc");
    }
}
