from email.message import EmailMessage
from email.utils import format_datetime
from datetime import datetime, timezone, timedelta
from pathlib import Path
import mailbox
import json

from analysis_pipeline import run_analysis

root = Path.cwd()
mbox_path = root / "mock_security.mbox"

# Recreate mock mbox each run for deterministic demo data.
if mbox_path.exists():
    mbox_path.unlink()

mbox = mailbox.mbox(str(mbox_path))
mbox.lock()

now = datetime.now(timezone.utc)

samples = [
    {
        "from": "Google <no-reply@accounts.google.com>",
        "subject": "중요 보안 알림: 새로운 로그인 감지",
        "body": "보안 경고입니다. 새 기기에서 로그인 시도가 감지되었습니다. 계정 보호를 위해 비밀번호 재설정을 권장합니다.",
        "date": now - timedelta(days=15),
    },
    {
        "from": "Naver <notice@naver.com>",
        "subject": "보안 알림 - 비밀번호 변경 안내",
        "body": "고객님의 계정 보안을 위해 비밀번호 변경을 완료했습니다. 본인이 아니라면 즉시 조치하세요.",
        "date": now - timedelta(days=40),
    },
    {
        "from": "Microsoft <account-security-noreply@accountprotection.microsoft.com>",
        "subject": "계정 보안 경고",
        "body": "보안 검토가 필요합니다. 의심스러운 로그인과 인증 실패가 반복되었습니다.",
        "date": now - timedelta(days=120),
    },
    {
        "from": "Old Service <security@legacy.example.com>",
        "subject": "보안 공지 (오래된 메일)",
        "body": "이 메일은 오래된 보안 공지 메일입니다.",
        "date": now - timedelta(days=280),
    },
    {
        "from": "Shop <noreply@shop.example.com>",
        "subject": "주문이 완료되었습니다",
        "body": "결제가 완료되었습니다. 감사합니다.",
        "date": now - timedelta(days=5),
    },
]

for item in samples:
    msg = EmailMessage()
    msg["From"] = item["from"]
    msg["To"] = "me@example.com"
    msg["Subject"] = item["subject"]
    msg["Date"] = format_datetime(item["date"])
    msg.set_content(item["body"])
    mbox.add(msg)

mbox.flush()
mbox.unlock()
mbox.close()

result = run_analysis(mbox_path=mbox_path, keywords=["보안"])

print(json.dumps(result.get("random_sender_report", {}), ensure_ascii=False, indent=2))
for row in result.get("sender_status_top3", []):
    print(json.dumps(row, ensure_ascii=False))
