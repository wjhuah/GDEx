"""Display actual sense experiments without presenting derived labels as gold."""

import json, html
from .ui import shell


def page(config):
    esc = html.escape
    root = config.reports / "sense_v3"
    inventory = json.loads((root / "amir_words.json").read_text())
    experiment = json.loads((root / "augmentation.json").read_text())
    body = (
        '<section class="panel"><h2>Amir 五词：TT 匹配与义项审核</h2><p>完整 token、lemma 与原生 xpos 等值匹配；不会把词内子串当成独立 token。同词性不同义项仍需语境判断，表中候选数不是正确率。</p><p>训练规模：'
        + esc(str(experiment["inventory"]))
        + "</p></section>"
    )
    body += '<section class="panel"><h3>逐词逐义项</h3><div class="table-scroll"><table><tr><th>方言</th><th>TLA / 词形</th><th>释义</th><th>严格 TT 候选</th><th>原人工正例文本</th></tr>'
    for r in inventory:
        body += f'<tr><td>{esc(r["dialect"])}</td><td><a href="/search?query={esc(r["tla"])}&dialect={esc(r["dialect"])}&ranker=research">{esc(r["tla"])} · {esc(r["lemma"])}</a></td><td>{esc(r["gloss"])}</td><td>{r["strict_TT_count"]}</td><td>{r["manual_positive_texts"]}</td></tr>'
    body += "</table></div><p>C771/C772 都是 ox/cow，不自动互作负例。ⲧⲱⲙⲧ 的两个动词义项仍共有同一 TT 候选池；不能说已经消歧。很多词没有独立人工义项测试。</p></section>"
    body += '<section class="panel"><h3>数据增强：真实五折结果</h3><p>按作品留出，清除重复原文；验证只用原人工收录/禁用标签。此处 AUC 是收录任务，不是五词的 token 义项准确率。</p><table><tr><th>方言</th><th>设置</th><th>有效折</th><th>平均 AUC</th></tr>'
    for d in ["sahidic", "bohairic"]:
        for mode in [
            "none",
            "naive_all",
            "cross",
            "cross_random",
            "cross_random_capped",
        ]:
            rows = [
                r
                for r in experiment["folds"]
                if r["dialect"] == d and r["mode"] == mode and r["auc"] is not None
            ]
            avg = f'{sum(r["auc"] for r in rows)/len(rows):.4f}' if rows else "无法评估"
            body += f"<tr><td>{d}</td><td>{mode}</td><td>{len(rows)}</td><td>{avg}</td></tr>"
    body += "</table><p>none：无增广；naive_all：不同释义同形词直接互作推导负例；cross：有 token/词性或词面证据的交叉负例；cross_random：再加不含目标 token 的随机负例；capped：总增广不超过目标正例数量。均为研究对照，不把生成标签当新专家标注。</p></section>"
    body += '<section class="panel"><h3>权重到底表示什么</h3><p>mean_word_frequency 是输入特征：本轮用排除目标后的上下文实词 token 平均词频。mean_log_word_frequency 是先取对数再平均。gain 是训练分裂带来的目标改善总量，split 是使用次数，均不是有正负号的线性权重。</p><p>默认45列研究树没有 cosine。原 Crum3 的 BGE 模型包含它，训练 gain 中它约占19.2%、排第一。不能混读两个模型，也不能把训练 gain 当作因果贡献。</p><p>句长检验同时删除 sent_len 和整组长度代理；只删 sent_len 还有 n_tokens 可以替代。详见当前消融页的两个新设置。</p><p><a href="/reports/sense_v3/REPORT.md">完整记录</a> · <a href="/reports/sense_v3/cdo_comparison.json">原词典逐条对照</a> · <a href="/reports/sense_v3/importance.json">各模型实际重要性</a></p></section>'
    return shell(body, "Sense audit", "sense")
