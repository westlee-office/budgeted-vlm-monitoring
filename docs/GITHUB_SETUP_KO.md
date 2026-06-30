# GitHub Setup

현재 상태:

- 로컬 git repository는 초기화되어 있다.
- initial commit은 생성되어 있다.
- raw dataset, feature cache, results, generated PDFs는 `.gitignore`로 제외된다.
- GitHub 원격 저장소 생성과 push는 외부 업로드이므로 명시 승인이 필요하다.

권장 remote 설정:

- repo name: `budgeted-vlm-monitoring`
- visibility: private
- account: `westlee-office`

명시 승인 후 실행할 명령:

```bash
gh repo create budgeted-vlm-monitoring \
  --private \
  --source=. \
  --remote=origin \
  --push
```

검증:

```bash
git remote -v
git status --short
gh repo view --web
```

이미 GitHub에서 repo를 먼저 만든 경우:

```bash
git remote add origin https://github.com/<owner>/budgeted-vlm-monitoring.git
git push -u origin main
```
