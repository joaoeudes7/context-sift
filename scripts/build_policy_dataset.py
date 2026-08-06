#!/usr/bin/env python3
"""Build deterministic multilingual policy KEEP/DROP supervision."""

from __future__ import annotations

import json
from pathlib import Path


LANGUAGES = {
    "en": {
        "goal": "Goal: reduce context without losing operational meaning.",
        "rule": "Never commit, tag, push, publish, or release without explicit user approval.",
        "permission": "Read-only repository inspection is allowed.",
        "prohibition": "Do not read .env, credentials, tokens, or private keys.",
        "precedence": "Project security rules override style preferences.",
        "request": "User request: update README installation instructions.",
        "question": "User question: can another service reuse the loaded model?",
        "command": "Update README installation instructions.",
        "sequence": "Bake cake -> sell cake -> get money -> repeat.",
        "noise": "The agent previously repeated a general reminder without adding a new requirement.",
    },
    "pt": {
        "goal": "Objetivo: reduzir contexto sem perder significado operacional.",
        "rule": "Nunca faça commit, tag, push, publicação ou release sem aprovação explícita do usuário.",
        "permission": "Inspeção somente leitura do repositório é permitida.",
        "prohibition": "Não leia .env, credenciais, tokens ou chaves privadas.",
        "precedence": "Regras de segurança do projeto prevalecem sobre preferências de estilo.",
        "request": "Pedido do usuário: atualize as instruções de instalação no README.",
        "question": "Pergunta do usuário: outro serviço pode reutilizar o modelo carregado?",
        "command": "Atualize as instruções de instalação no README.",
        "sequence": "Fazer bolo -> vender bolo -> obter dinheiro -> repetir.",
        "noise": "O agente repetiu anteriormente um lembrete geral sem adicionar requisito novo.",
    },
    "es": {
        "goal": "Objetivo: reducir el contexto sin perder significado operativo.",
        "rule": "Nunca haga commit, tag, push, publicación ni release sin aprobación explícita del usuario.",
        "permission": "Se permite la inspección de solo lectura del repositorio.",
        "prohibition": "No lea .env, credenciales, tokens ni claves privadas.",
        "precedence": "Las reglas de seguridad del proyecto prevalecen sobre preferencias de estilo.",
        "request": "Solicitud del usuario: actualice las instrucciones de instalación del README.",
        "question": "Pregunta del usuario: ¿otro servicio puede reutilizar el modelo cargado?",
        "command": "Actualiza las instrucciones de instalación del README.",
        "sequence": "Hacer pastel -> vender pastel -> obtener dinero -> repetir.",
        "noise": "El agente repitió un recordatorio general sin agregar un requisito nuevo.",
    },
    "fr": {
        "goal": "Objectif : réduire le contexte sans perdre le sens opérationnel.",
        "rule": "Ne créez jamais de commit, tag, push, publication ou release sans accord explicite.",
        "permission": "L'inspection du dépôt en lecture seule est autorisée.",
        "prohibition": "Ne lisez pas .env, les identifiants, jetons ou clés privées.",
        "precedence": "Les règles de sécurité du projet priment sur les préférences de style.",
        "request": "Demande utilisateur : mettez à jour les instructions d'installation du README.",
        "question": "Question utilisateur : un autre service peut-il réutiliser le modèle chargé ?",
        "command": "Mettez à jour les instructions d'installation du README.",
        "sequence": "Faire un gâteau -> vendre le gâteau -> obtenir de l'argent -> répéter.",
        "noise": "L'agent a répété un rappel général sans ajouter de nouvelle exigence.",
    },
    "de": {
        "goal": "Ziel: Kontext reduzieren, ohne operative Bedeutung zu verlieren.",
        "rule": "Niemals Commit, Tag, Push, Veröffentlichung oder Release ohne ausdrückliche Zustimmung.",
        "permission": "Eine schreibgeschützte Repository-Prüfung ist erlaubt.",
        "prohibition": "Lies niemals .env, Zugangsdaten, Tokens oder private Schlüssel.",
        "precedence": "Projektsicherheitsregeln haben Vorrang vor Stilpräferenzen.",
        "request": "Benutzerauftrag: Aktualisiere die Installationsanweisungen im README.",
        "question": "Benutzerfrage: Kann ein anderer Dienst das geladene Modell wiederverwenden?",
        "command": "Aktualisiere die Installationsanweisungen im README.",
        "sequence": "Kuchen backen -> Kuchen verkaufen -> Geld erhalten -> wiederholen.",
        "noise": "Der Agent wiederholte einen allgemeinen Hinweis ohne neue Anforderung.",
    },
    "ru": {
        "goal": "Цель: сократить контекст без потери операционного смысла.",
        "rule": "Никогда не выполняйте commit, tag, push, публикацию или release без явного одобрения.",
        "permission": "Разрешена проверка репозитория только для чтения.",
        "prohibition": "Не читайте .env, учетные данные, токены или закрытые ключи.",
        "precedence": "Правила безопасности проекта важнее стилевых предпочтений.",
        "request": "Запрос пользователя: обновите инструкции по установке в README.",
        "question": "Вопрос пользователя: может ли другой сервис повторно использовать загруженную модель?",
        "command": "Обнови инструкции по установке в README.",
        "sequence": "Испечь торт -> продать торт -> получить деньги -> повторить.",
        "noise": "Агент повторил общее напоминание без нового требования.",
    },
    "ar": {
        "goal": "الهدف: تقليل السياق دون فقدان المعنى التشغيلي.",
        "rule": "لا تنفذ commit أو tag أو push أو نشر أو release دون موافقة صريحة من المستخدم.",
        "permission": "يسمح بفحص المستودع للقراءة فقط.",
        "prohibition": "لا تقرأ .env أو بيانات الاعتماد أو الرموز أو المفاتيح الخاصة.",
        "precedence": "قواعد أمان المشروع تتقدم على تفضيلات الأسلوب.",
        "request": "طلب المستخدم: حدّث تعليمات التثبيت في README.",
        "question": "سؤال المستخدم: هل يمكن لخدمة أخرى إعادة استخدام النموذج المحمل؟",
        "command": "حدّث تعليمات التثبيت في README.",
        "sequence": "اصنع كعكة -> بع الكعكة -> احصل على المال -> كرر.",
        "noise": "كرر الوكيل تذكيرا عاما دون إضافة متطلب جديد.",
    },
    "ja": {
        "goal": "目的: 運用上の意味を失わずにコンテキストを削減する。",
        "rule": "ユーザーの明示的な承認なしに commit、tag、push、公開、release を実行してはいけない。",
        "permission": "リポジトリの読み取り専用調査は許可される。",
        "prohibition": ".env、認証情報、トークン、秘密鍵を読んではいけない。",
        "precedence": "プロジェクトのセキュリティ規則はスタイル設定より優先される。",
        "request": "ユーザーの依頼: README のインストール手順を更新する。",
        "question": "ユーザーの質問: 別のサービスが読み込み済みモデルを再利用できるか？",
        "command": "README のインストール手順を更新する。",
        "sequence": "ケーキを作る -> 売る -> お金を得る -> 繰り返す。",
        "noise": "エージェントは新しい要件を加えず一般的な注意を繰り返した。",
    },
    "zh": {
        "goal": "目标：在不丢失操作含义的情况下缩短上下文。",
        "rule": "未经用户明确批准，不得执行 commit、tag、push、发布或 release。",
        "permission": "允许对仓库进行只读检查。",
        "prohibition": "不得读取 .env、凭据、令牌或私钥。",
        "precedence": "项目安全规则优先于风格偏好。",
        "request": "用户请求：更新 README 中的安装说明。",
        "question": "用户问题：其他服务能否复用已加载的模型？",
        "command": "更新 README 中的安装说明。",
        "sequence": "制作蛋糕 -> 出售蛋糕 -> 获得金钱 -> 重复。",
        "noise": "代理重复了一般提醒，但没有增加新要求。",
    },
}

EN_COMMANDS = (
    "Do something useful with the repository.",
    "You should update README installation instructions.",
    "Do you can continue the validation?",
    "Search by exact issue identifier AUTH-417.",
    "Keep running the regression tests.",
    "Find the relevant implementation file.",
    "Identify the root cause before editing.",
)


def row(language: str, values: dict[str, str], index: int) -> dict:
    command = EN_COMMANDS[index % len(EN_COMMANDS)] if language == "en" else values["command"]
    keep = [
        values["goal"], values["request"], values["question"], command, values["sequence"], values["rule"],
        values["permission"], values["prohibition"], values["precedence"],
    ]
    noise = [values["noise"]] * (3 + index % 5)
    duplicate = values["rule"]
    units = keep + noise + [duplicate]
    shift = index % len(units)
    units = units[shift:] + units[:shift]
    seen_rule = False
    labels = []
    for text in units:
        if text == values["noise"]:
            labels.append(0)
        elif text == duplicate and seen_rule:
            labels.append(0)
        else:
            labels.append(1)
            if text == duplicate:
                seen_rule = True
    return {
        "id": f"policy-{language}-{index}",
        "group_id": f"policy-{language}-{index}",
        "language": language,
        "source": "\n".join(units),
        "units": [{"id": f"u{i}", "text": text} for i, text in enumerate(units)],
        "target_unit_ids": [f"u{i}" for i, label in enumerate(labels) if label],
        "critical": keep,
    }


def write(path: Path, start: int, count: int) -> None:
    rows = [row(language, values, index) for language, values in LANGUAGES.items() for index in range(start, start + count)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows), encoding="utf-8")
    print(json.dumps({"path": str(path), "rows": len(rows)}, ensure_ascii=False))


def main() -> None:
    write(Path("data/msc/policy_rules.jsonl"), 0, 40)
    write(Path("data/validation/prompts/policy_rules.jsonl"), 100, 5)


if __name__ == "__main__":
    main()
