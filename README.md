# taxi

## 로컬에서 실행하기

1. `uv`가 없다면 설치합니다.

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. 저장소 폴더에서 환경을 준비합니다. 최초 1회만 실행하면 됩니다.

   ```bash
   uv sync
   ```

3. Jupyter를 실행합니다.

   ```bash
   uv run jupyter lab main.ipynb
   ```

다음부터는 3번 명령만 실행하면 됩니다.

실제 분석 전 `data/raw/yellow_tripdata_2026-05.csv`를 사용할 데이터로 교체하세요.
