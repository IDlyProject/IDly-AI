from __future__ import annotations

import html
import hashlib
import mailbox
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from email.header import decode_header
from email.utils import parseaddr
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

try:
    import hdbscan
except Exception:  # pragma: no cover
    hdbscan = None


DEFAULT_STOPWORDS = {
    "안녕하세요",
    "고객님",
    "감사합니다",
    "문의",
    "확인",
    "이용",
    "관련",
    "안내",
    "서비스",
    "회원",
    "정보",
    "부탁드립니다",
    "바랍니다",
    "드립니다",
    "입니다",
    "되었습니다",
    "가능합니다",
    "the",
    "and",
    "for",
    "from",
    "that",
    "this",
    "with",
    "your",
    "you",
    "have",
    "has",
    "will",
    "was",
    "are",
    "not",
    "please",
    "mail",
    "email",
    "account",
    "login",
    "px",
    "font",
    "display",
    "width",
    "margin",
    "color",
    "background",
    "class",
    "important",
    "media",
    "block",
    "normal",
    "center",
    "left",
    "right",
    "all",
    "max",
    "min",
    "img",
    "src",
    "td",
    "tr",
    "th",
    "tbody",
    "thead",
    "table",
    "span",
    "div",
    "nbsp",
    "href",
    "http",
    "https",
    "roboto",
    "arial",
    "sans",
    "serif",
    "padding",
    "border",
    "radius",
    "height",
    "family",
    "align",
    "mso",
    "woff",
    "woff2",
    "format",
    "local",
    "top",
    "bottom",
    "row",
    "col",
    "ib",
    "mb",
    "flex",
}


@dataclass
class AnalysisConfig:
    mail_tfidf_max_features: int = 2000
    sender_tfidf_max_features: int = 1500
    min_df_mail: int = 2
    max_df_mail: float = 0.85
    max_df_sender: float = 0.9
    ngram_range: tuple[int, int] = (1, 2)
    max_clusters: int = 6
    min_clusters: int = 3
    hdbscan_min_cluster_size: int = 20
    hdbscan_min_samples: int = 5
    isolation_contamination: float = 0.08
    random_state: int = 42
    enable_clustering: bool = False
    enable_bertopic: bool = False


class VisibleTextParser(HTMLParser):
    skip_tags = {"script", "style", "head", "title", "meta", "noscript"}

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in self.skip_tags:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self.skip_tags and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data and data.strip():
            self._parts.append(data.strip())

    def get_text(self) -> str:
        return " ".join(self._parts)


def decode_mime_words(value: str | None) -> str:
    if not value:
        return ""

    decoded: list[str] = []
    for part, enc in decode_header(value):
        if isinstance(part, bytes):
            for candidate in [enc, "utf-8", "cp949", "euc-kr", "latin1"]:
                if not candidate:
                    continue
                try:
                    decoded.append(part.decode(candidate, errors="replace"))
                    break
                except Exception:
                    continue
            else:
                decoded.append(part.decode("utf-8", errors="replace"))
        else:
            decoded.append(part)
    return "".join(decoded)


def html_to_visible_text(raw_html: str) -> str:
    raw_html = raw_html[:300000]
    raw_html = re.sub(r"<!--.*?-->", " ", raw_html, flags=re.S)

    parser = VisibleTextParser()
    parser.feed(raw_html)
    parser.close()

    text = html.unescape(parser.get_text())
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[\w\.-]+@[\w\.-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_text_body(msg: Any) -> str:
    chunks: list[str] = []
    parts = msg.walk() if msg.is_multipart() else [msg]

    for part in parts:
        content_type = part.get_content_type()
        disposition = str(part.get("Content-Disposition", ""))

        if "attachment" in disposition.lower():
            continue
        if content_type not in ["text/plain", "text/html"]:
            continue

        payload = part.get_payload(decode=True)
        if not payload:
            continue

        charset = part.get_content_charset() or "utf-8"
        try:
            text = payload.decode(charset, errors="replace")
        except Exception:
            text = payload.decode("utf-8", errors="replace")

        if content_type == "text/html":
            text = html_to_visible_text(text)
        else:
            text = html.unescape(text)
            text = re.sub(r"\s+", " ", text).strip()

        if text:
            chunks.append(text[:5000])

    return "\n".join(chunks)


def extract_keyword_matches(mbox_path: str | Path, keywords: list[str]) -> list[dict[str, Any]]:
    matched: list[dict[str, Any]] = []
    mbox = mailbox.mbox(str(mbox_path))

    for i, msg in enumerate(mbox):
        subject = decode_mime_words(msg.get("Subject", ""))
        sender_raw = decode_mime_words(msg.get("From", ""))
        sender_name, sender_email = parseaddr(sender_raw)
        sender_email = sender_email.lower().strip()
        sender_domain = sender_email.split("@")[-1] if "@" in sender_email else ""
        sender = f"{sender_name} <{sender_email}>" if sender_email else sender_raw
        date = decode_mime_words(msg.get("Date", ""))
        body = extract_text_body(msg)

        searchable = f"{subject}\n{body}"
        found_keywords = [kw for kw in keywords if kw in searchable]

        if found_keywords:
            matched.append(
                {
                    "index": i,
                    "sender": sender,
                    "sender_email": sender_email,
                    "sender_domain": sender_domain,
                    "subject": subject,
                    "date": date,
                    "body": body,
                    "matched_keywords": ", ".join(found_keywords),
                }
            )

    return matched


def preprocess_text(text: str, stopwords: set[str]) -> tuple[str, list[str]]:
    text = (text or "").lower()
    text = html.unescape(text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[\w\.-]+@[\w\.-]+", " ", text)
    text = re.sub(r"[^0-9a-zA-Z가-힣\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = re.findall(r"[가-힣]{2,}|[a-z]{2,}|\d{2,}", text)
    tokens = [
        token
        for token in tokens
        if token not in stopwords
        and len(token) > 1
        and not re.fullmatch(r"\d+(px|pt|em|rem)", token)
    ]
    return text, tokens


def preprocess_records(
    matched: list[dict[str, Any]],
    stopwords: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[tuple[str, int]]], Counter[str]]:
    stopwords = stopwords or DEFAULT_STOPWORDS
    preprocessed_records: list[dict[str, Any]] = []
    sender_mail_counts: Counter[str] = Counter()
    sender_token_bags: defaultdict[str, list[str]] = defaultdict(list)

    for item in matched:
        combined_text = f"{item['subject']}\n{item['body']}"
        clean_text, tokens = preprocess_text(combined_text, stopwords)

        sender_key = item.get("sender_email") or item.get("sender") or "unknown_sender"
        sender_mail_counts[sender_key] += 1
        sender_token_bags[sender_key].extend(tokens)

        preprocessed_records.append(
            {
                **item,
                "clean_text": clean_text,
                "tokens": tokens,
                "token_count": len(tokens),
            }
        )

    sender_profiles = {
        sender: Counter(tokens).most_common(15)
        for sender, tokens in sender_token_bags.items()
    }
    return preprocessed_records, sender_profiles, sender_mail_counts


def build_tfidf(
    preprocessed_records: list[dict[str, Any]],
    config: AnalysisConfig,
    include_mail: bool = False,
    include_sender: bool = True,
) -> dict[str, Any]:
    if not preprocessed_records:
        raise ValueError("No preprocessed records available.")

    mail_tfidf_matrix = None
    mail_feature_names = None
    if include_mail:
        mail_documents = [
            " ".join(row["tokens"]) if row["tokens"] else row["clean_text"]
            for row in preprocessed_records
        ]

        mail_vectorizer = TfidfVectorizer(
            max_features=config.mail_tfidf_max_features,
            min_df=config.min_df_mail,
            max_df=config.max_df_mail,
            ngram_range=config.ngram_range,
        )
        mail_tfidf_matrix = mail_vectorizer.fit_transform(mail_documents)
        mail_feature_names = mail_vectorizer.get_feature_names_out()

    sender_tfidf_matrix = None
    sender_feature_names = None
    sender_ids: list[str] = []
    if include_sender:
        sender_documents: dict[str, list[str]] = {}
        for row in preprocessed_records:
            sender_key = row.get("sender_email") or row.get("sender") or "unknown_sender"
            sender_documents.setdefault(sender_key, [])
            sender_documents[sender_key].append(" ".join(row["tokens"]))

        sender_ids = list(sender_documents.keys())
        sender_corpus = [" ".join(sender_documents[sender]) for sender in sender_ids]

        sender_vectorizer = TfidfVectorizer(
            max_features=config.sender_tfidf_max_features,
            min_df=1,
            max_df=config.max_df_sender,
            ngram_range=config.ngram_range,
        )
        sender_tfidf_matrix = sender_vectorizer.fit_transform(sender_corpus)
        sender_feature_names = sender_vectorizer.get_feature_names_out()

    return {
        "mail_tfidf_matrix": mail_tfidf_matrix,
        "mail_feature_names": mail_feature_names,
        "sender_tfidf_matrix": sender_tfidf_matrix,
        "sender_feature_names": sender_feature_names,
        "sender_ids": sender_ids,
    }


def cluster_mail(
    mail_tfidf_matrix: Any,
    preprocessed_records: list[dict[str, Any]],
    config: AnalysisConfig,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None]:
    mail_docs_count = len(preprocessed_records)
    if mail_docs_count < 10:
        raise ValueError("Need at least 10 mails to run clustering.")

    n_components = min(20, max(2, mail_tfidf_matrix.shape[1] - 1))
    svd_for_cluster = TruncatedSVD(
        n_components=n_components,
        random_state=config.random_state,
    )
    mail_reduced = svd_for_cluster.fit_transform(mail_tfidf_matrix)
    mail_2d = mail_reduced[:, :2]

    n_clusters = min(
        config.max_clusters,
        max(config.min_clusters, mail_docs_count // 700 + config.min_clusters),
    )
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=config.random_state,
        n_init=10,
    )
    kmeans_labels = kmeans.fit_predict(mail_reduced)

    hdbscan_labels: np.ndarray | None = None
    if hdbscan is not None:
        hdb = hdbscan.HDBSCAN(
            min_cluster_size=config.hdbscan_min_cluster_size,
            min_samples=config.hdbscan_min_samples,
            metric="euclidean",
        )
        hdbscan_labels = hdb.fit_predict(mail_reduced)

    cluster_results_df = pd.DataFrame(
        {
            "sender": [row["sender"] for row in preprocessed_records],
            "subject": [row["subject"] for row in preprocessed_records],
            "matched_keywords": [row["matched_keywords"] for row in preprocessed_records],
            "kmeans_cluster": kmeans_labels,
            "hdbscan_cluster": hdbscan_labels if hdbscan_labels is not None else -1,
            "x": mail_2d[:, 0],
            "y": mail_2d[:, 1],
        }
    )
    return cluster_results_df, kmeans_labels, hdbscan_labels


def topic_and_anomaly(
    preprocessed_records: list[dict[str, Any]],
    sender_tfidf_matrix: Any,
    sender_ids: list[str],
    sender_mail_counts: Counter[str],
    stopwords: set[str],
    config: AnalysisConfig,
) -> tuple[list[int], pd.DataFrame, pd.DataFrame]:
    topics = [-1] * len(preprocessed_records)
    topic_info = pd.DataFrame(columns=["Topic", "Count", "Name"])
    BERTopic = None

    if config.enable_bertopic and preprocessed_records:
        try:
            from bertopic import BERTopic  # Lazy import for performance.
        except Exception:
            BERTopic = None

    if config.enable_bertopic and BERTopic is not None and preprocessed_records:
        topic_docs = [
            row["clean_text"] if row["clean_text"] else " ".join(row["tokens"])
            for row in preprocessed_records
        ]
        vectorizer_model = CountVectorizer(
            stop_words=list(stopwords),
            ngram_range=config.ngram_range,
            min_df=2,
        )
        topic_model = BERTopic(
            embedding_model=None,
            vectorizer_model=vectorizer_model,
            min_topic_size=20,
            calculate_probabilities=False,
            verbose=False,
        )
        try:
            topics, _ = topic_model.fit_transform(topic_docs)
            topic_info = topic_model.get_topic_info()
        except Exception:
            # Small/edge datasets may fail UMAP reduction in BERTopic.
            topics = [-1] * len(preprocessed_records)
            topic_info = pd.DataFrame(columns=["Topic", "Count", "Name"])

    sender_dense = sender_tfidf_matrix.toarray()
    iso_forest = IsolationForest(
        contamination=config.isolation_contamination,
        random_state=config.random_state,
    )
    iso_pred = iso_forest.fit_predict(sender_dense)
    anomaly_score = -iso_forest.decision_function(sender_dense)

    sender_anomaly_df = pd.DataFrame(
        {
            "sender_id": sender_ids,
            "anomaly_flag": iso_pred,
            "anomaly_score": anomaly_score,
            "mail_count": [sender_mail_counts.get(sender, 0) for sender in sender_ids],
        }
    ).sort_values("anomaly_score", ascending=False)

    return topics, topic_info, sender_anomaly_df


def infer_risk_level(score: float) -> str:
    if score >= 5:
        return "높음"
    if score >= 2.5:
        return "중간"
    return "낮음"


def infer_account_state(row: pd.Series) -> str:
    if row["avg_reset_signal"] >= 1.2:
        return "복구/비밀번호 재설정"
    if row["avg_login_signal"] >= 1.5 and row["avg_alert_signal"] >= 1.0:
        return "로그인 이상 탐지"
    if row["avg_action_signal"] >= 1.0 or row["avg_alert_signal"] >= 2.0:
        return "보안 조치 필요"
    if row["anomaly_score"] >= 0.08 and row["mail_count"] >= 3:
        return "이상 패턴 감지"
    if row["avg_login_signal"] >= 1.0:
        return "로그인/인증 활동"
    return "일반 보안 알림"


def estimate_sender_status(
    preprocessed_records: list[dict[str, Any]],
    sender_anomaly_df: pd.DataFrame,
    kmeans_labels: np.ndarray | None = None,
    topics: list[int] | None = None,
    topic_info: pd.DataFrame | None = None,
) -> pd.DataFrame:
    sender_analysis_df = pd.DataFrame(preprocessed_records).copy()
    sender_analysis_df["sender_key"] = sender_analysis_df["sender_email"].fillna("")
    sender_analysis_df["sender_key"] = sender_analysis_df["sender_key"].where(
        sender_analysis_df["sender_key"] != "",
        sender_analysis_df["sender"],
    )

    signal_terms = {
        "reset_signal": ["비밀번호", "재설정", "reset", "recover", "recovery", "복구", "password"],
        "login_signal": ["로그인", "signin", "sign in", "인증", "코드", "otp", "verify", "verification", "로그인되었습니다", "새로 로그인"],
        "alert_signal": ["보안", "security", "알림", "경고", "의심", "위험", "suspicious", "차단", "잠금", "잠김", "보호"],
        "action_signal": ["조치", "권장", "확인", "검토", "review", "변경", "update", "필수", "승인", "해제"],
    }

    def count_signal_hits(text: str, terms: list[str]) -> int:
        text = (text or "").lower()
        return sum(text.count(term.lower()) for term in terms)

    for col_name, terms in signal_terms.items():
        sender_analysis_df[col_name] = sender_analysis_df["clean_text"].apply(
            lambda x: count_signal_hits(x, terms)
        )

    if kmeans_labels is not None and len(kmeans_labels) == len(sender_analysis_df):
        sender_analysis_df["kmeans_cluster"] = kmeans_labels
    else:
        sender_analysis_df["kmeans_cluster"] = -1

    if topics is not None and len(topics) == len(sender_analysis_df):
        sender_analysis_df["topic_id"] = topics
    else:
        sender_analysis_df["topic_id"] = -999

    topic_name_map: dict[int, str] = {}
    if topic_info is not None and isinstance(topic_info, pd.DataFrame) and not topic_info.empty:
        topic_name_map = dict(zip(topic_info["Topic"], topic_info["Name"]))

    def most_common_value(series: pd.Series) -> Any:
        series = pd.Series(series)
        if series.empty:
            return None
        return series.value_counts().idxmax()

    sender_status_df = (
        sender_analysis_df.groupby("sender_key")
        .agg(
            sender_name=("sender", "first"),
            mail_count=("sender_key", "size"),
            sample_subject=("subject", "first"),
            reset_signal=("reset_signal", "sum"),
            login_signal=("login_signal", "sum"),
            alert_signal=("alert_signal", "sum"),
            action_signal=("action_signal", "sum"),
            dominant_cluster=("kmeans_cluster", most_common_value),
            dominant_topic=("topic_id", most_common_value),
        )
        .reset_index()
    )

    sender_status_df = sender_status_df.merge(
        sender_anomaly_df[["sender_id", "anomaly_flag", "anomaly_score"]],
        left_on="sender_key",
        right_on="sender_id",
        how="left",
    ).drop(columns=["sender_id"])

    sender_status_df["anomaly_score"] = sender_status_df["anomaly_score"].fillna(0.0)
    sender_status_df["anomaly_flag"] = sender_status_df["anomaly_flag"].fillna(1)

    for metric in ["reset_signal", "login_signal", "alert_signal", "action_signal"]:
        sender_status_df[f"avg_{metric}"] = sender_status_df[metric] / sender_status_df[
            "mail_count"
        ].clip(lower=1)

    sender_status_df["dominant_topic_name"] = (
        sender_status_df["dominant_topic"].map(topic_name_map).fillna("unknown")
    )

    sender_status_df["risk_score"] = (
        sender_status_df["avg_alert_signal"] * 2.2
        + sender_status_df["avg_action_signal"] * 1.8
        + sender_status_df["avg_reset_signal"] * 1.5
        + sender_status_df["avg_login_signal"] * 1.2
        + sender_status_df["anomaly_score"] * 8
    )
    sender_status_df["risk_level"] = sender_status_df["risk_score"].apply(infer_risk_level)
    sender_status_df["account_state"] = sender_status_df.apply(infer_account_state, axis=1)

    sender_status_df = sender_status_df.sort_values(
        ["risk_score", "mail_count"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return sender_status_df


def score_to_state(score: float) -> str:
    if score >= 5:
        return "위험"
    if score >= 2.5:
        return "주의"
    return "양호"


def build_account_summary(
    sender_status_df: pd.DataFrame,
    preprocessed_records: list[dict[str, Any]],
) -> dict[str, Any]:
    if sender_status_df.empty:
        return {"accounts": []}

    state_reason_map = {
        "복구/비밀번호 재설정": "비밀번호 재설정, 계정 복구, 인증 복원 관련 표현이 반복적으로 나타났습니다.",
        "로그인 이상 탐지": "새 로그인, 인증 코드, 본인 확인, 보안 알림이 함께 관찰되었습니다.",
        "보안 조치 필요": "경고/조치/검토 요청 표현이 누적되어 후속 확인이 필요한 상태입니다.",
        "이상 패턴 감지": "메일 패턴이 다른 발신자에 비해 다르게 나타나 추가 확인이 필요합니다.",
        "로그인/인증 활동": "로그인 및 인증 관련 활동이 중심인 발신자입니다.",
        "일반 보안 알림": "주로 안내성 보안 알림 메일이 관찰됩니다.",
    }

    sender_problem_mails: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in preprocessed_records:
        sender_key = row.get("sender_email") or row.get("sender") or "unknown_sender"
        sender_problem_mails[sender_key].append(
            {
                "subject": row.get("subject", ""),
                "date": row.get("date", ""),
                "body": row.get("body", ""),
                "matched_keywords": row.get("matched_keywords", ""),
            }
        )

    accounts: list[dict[str, Any]] = []
    for _, row in sender_status_df.iterrows():
        score = float(row.get("risk_score", 0.0))
        account_state = row.get("account_state", "일반 보안 알림")
        sender_key = str(row.get("sender_key", "unknown_sender"))
        account_id = f"acct_{hashlib.sha1(sender_key.encode('utf-8')).hexdigest()[:12]}"
        accounts.append(
            {
                "account_id": account_id,
                "account": row.get("sender_name") or row.get("sender_key"),
                "security_score": round(score, 3),
                "security_level": score_to_state(score),
                "interpretation": state_reason_map.get(
                    account_state,
                    "보안 관련 메일 패턴을 바탕으로 추정했습니다.",
                ),
                "problem_mails": sender_problem_mails.get(sender_key, []),
            }
        )

    return {"accounts": accounts}


def run_analysis(
    mbox_path: str | Path,
    keywords: list[str],
    config: AnalysisConfig | None = None,
    stopwords: set[str] | None = None,
) -> dict[str, Any]:
    config = config or AnalysisConfig()
    stopwords = stopwords or DEFAULT_STOPWORDS

    matched = extract_keyword_matches(mbox_path=mbox_path, keywords=keywords)
    if not matched:
        return {"accounts": []}

    preprocessed_records, sender_profiles, sender_mail_counts = preprocess_records(
        matched,
        stopwords,
    )
    tfidf_data = build_tfidf(
        preprocessed_records,
        config,
        include_mail=config.enable_clustering,
        include_sender=True,
    )

    kmeans_labels: np.ndarray | None = None
    hdbscan_labels: np.ndarray | None = None
    cluster_results_df: pd.DataFrame | None = None
    if (
        config.enable_clustering
        and len(preprocessed_records) >= 10
        and tfidf_data["mail_tfidf_matrix"] is not None
    ):
        cluster_results_df, kmeans_labels, hdbscan_labels = cluster_mail(
            tfidf_data["mail_tfidf_matrix"],
            preprocessed_records,
            config,
        )

    topics, topic_info, sender_anomaly_df = topic_and_anomaly(
        preprocessed_records=preprocessed_records,
        sender_tfidf_matrix=tfidf_data["sender_tfidf_matrix"],
        sender_ids=tfidf_data["sender_ids"],
        sender_mail_counts=sender_mail_counts,
        stopwords=stopwords,
        config=config,
    )

    sender_status_df = estimate_sender_status(
        preprocessed_records=preprocessed_records,
        sender_anomaly_df=sender_anomaly_df,
        kmeans_labels=kmeans_labels,
        topics=topics,
        topic_info=topic_info,
    )

    return build_account_summary(
        sender_status_df=sender_status_df,
        preprocessed_records=preprocessed_records,
    )
