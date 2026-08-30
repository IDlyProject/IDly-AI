#!/usr/bin/env python
"""프로파일링을 위한 테스트 스크립트"""
import logging
from pathlib import Path
from analysis_pipeline import run_analysis

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 작은 파일로 먼저 테스트
print("=" * 60)
print("Testing with mock.mbox (small)")
print("=" * 60)
result = run_analysis(
    mbox_path="mock.mbox",
    keywords=["보안"]
)
print(f"\nResult accounts: {len(result.get('accounts', []))}")

# 큰 파일로 테스트 (선택사항)
large_file = Path("hyeon05_ewha.ac.kr.mbox")
if large_file.exists():
    print("\n" + "=" * 60)
    print(f"Testing with {large_file.name} (large, ~63MB)")
    print("=" * 60)
    result = run_analysis(
        mbox_path=str(large_file),
        keywords=["보안"]
    )
    print(f"\nResult accounts: {len(result.get('accounts', []))}")
