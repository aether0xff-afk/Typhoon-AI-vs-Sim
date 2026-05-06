# Typhoon AI vs Sim

단순 물리 모델이 예측한 태풍 강도를 인공지능이 잔차(residual) 방식으로 보정하는 실험 프로젝트다.  
이제 같은 구조를 `합성 데이터`와 `실제 데이터(IBTrACS + NOAA OISST)` 두 경로에서 모두 실행할 수 있다.

핵심 구조는 다음과 같다.

1. 데이터 소스를 고른다.
2. 단순 물리 모델이 다음 시점 강도를 먼저 예측한다.
3. `MLP`, `LSTM`, `Transformer`가 물리 모델의 오차를 학습한다.
4. `물리 예측값 + AI 보정값`으로 최종 강도를 계산한다.
5. `MAE`, `RMSE`로 물리 단독 모델과 AI 보정 모델을 비교한다.

데이터 소스는 다음 둘 중 하나다.

- `synthetic`: 서태평양 태풍 규모를 흉내 낸 합성 시계열을 생성한다.
- `ibtracs`: NOAA IBTrACS 서태평양 best track과 NOAA OISST를 자동으로 내려받아 실제 태풍 시계열을 만든다.

이 구조는 다음 결론에 맞춰 설계되어 있다.

> 단순 물리 모델은 태풍 강도 변화의 기본 경향을 설명할 수 있었고, 인공지능 보정 모델을 결합했을 때 물리 모델의 예측 오차를 줄여 더 정확하게 태풍 강도 변화를 예측할 수 있었다.

## 설치

```bash
python -m pip install -r requirements.txt
```

## 실행

합성 데이터 빠른 스모크 실행:

```bash
python main.py --quick --output-dir outputs/smoke
```

합성 데이터 기본 검증 실행:

```bash
python main.py --output-dir outputs/run
```

합성 데이터 직접 크기 조절:

```bash
python main.py --storms 150 --epochs 50 --window-size 6 --output-dir outputs/custom
```

실데이터 스모크 실행:

```bash
python main.py --data-source ibtracs --ibtracs-start-year 2020 --ibtracs-end-year 2020 --max-real-storms 4 --epochs 5 --output-dir outputs/smoke_real
```

실데이터 기본 실행 예시:

```bash
python main.py --data-source ibtracs --ibtracs-start-year 2018 --ibtracs-end-year 2020 --max-real-storms 24 --output-dir outputs/real_run
```

실데이터 실행 시 `data/` 아래에 다음 캐시가 자동으로 저장된다.

- `ibtracs.WP.list.v04r01.csv`
- `oisst_cache/*.nc`

## 생성되는 결과물

실행이 끝나면 `output-dir` 아래에 다음 파일들이 저장된다.

- `storm_tracks.csv`: 사용된 태풍 시계열 원본 데이터
- `synthetic_storm_tracks.csv`: 합성 실험일 때만 저장되는 호환용 파일
- `data_source_summary.json`: 데이터 소스, 필터링 조건, 합성 파라미터 근거 또는 실데이터 출처 요약
- `predictions.csv`: 물리 모델과 AI 보정 모델의 예측값
- `metrics.csv`: `MAE`, `RMSE`, 개선율 비교
- `training_history.csv`: 학습 손실 기록
- `summary.json`: 핵심 설정과 테스트 메트릭 요약
- `metrics_comparison.png`: 모델별 오차 비교 그래프
- `sample_trajectories.png`: 실제 강도와 예측 강도 비교 그래프
- `training_curves.png`: 학습 곡선
- `checkpoints/*.pt`: 각 AI 모델의 최적 가중치

## 프로젝트 구조

```text
main.py
typhoon_ai_vs_sim/
  cli.py
  config.py
  data.py
  data_sources.py
  experiment.py
  models.py
  plotting.py
  simulation.py
  train.py
  utils.py
```

## 구현 포인트

- 물리 모델은 `SST`, 위도, 강도 포화 효과만 반영하는 단순한 규칙 기반 모델이다.
- 합성 데이터는 서태평양 태풍의 6시간 간격 진행을 흉내 내도록 설계했고, 파라미터 범위와 근거를 `data_source_summary.json`에 함께 저장한다.
- 실제 데이터는 NOAA IBTrACS 서태평양 트랙에서 강도를 읽고, NOAA OISST 일별 격자에서 태풍 중심 위치의 SST를 최근접 추출한다.
- 합성 기준 데이터는 급강화 구간, 습도, 전단, 기억 효과 같은 비선형 요소를 추가로 포함한다.
- AI는 강도 자체를 직접 예측하지 않고 물리 모델의 오차만 학습한다.
- 따라서 결과 해석을 `AI가 물리 모델을 보정했다`는 방향으로 깔끔하게 가져갈 수 있다.

## 검증 예시

`python main.py --storms 120 --epochs 40 --output-dir outputs/validation` 실행에서 다음과 같은 테스트 결과가 나왔다.

| Model | MAE | RMSE | MAE 개선율 |
| --- | ---: | ---: | ---: |
| Physics | 0.8503 | 1.0874 | 0.00% |
| MLP | 0.4137 | 0.5242 | 51.35% |
| LSTM | 0.3919 | 0.4962 | 53.91% |
| Transformer | 0.4378 | 0.5746 | 48.51% |

즉 이 실험 설정에서는 AI 보정 모델이 물리 모델보다 더 낮은 오차를 보였다.

실데이터 스모크 테스트는 다음 명령으로 확인했다.

```bash
python main.py --data-source ibtracs --ibtracs-start-year 2020 --ibtracs-end-year 2020 --max-real-storms 4 --epochs 5 --output-dir outputs/smoke_real
```

이 실행에서는 테스트셋 기준으로 다음 값이 나왔다.

| Model | MAE | RMSE | MAE 개선율 |
| --- | ---: | ---: | ---: |
| Physics | 3.1341 | 3.5783 | 0.00% |
| MLP | 2.7675 | 3.2481 | 11.70% |
| LSTM | 2.0391 | 2.3965 | 34.94% |
| Transformer | 1.9225 | 2.5483 | 38.66% |

표본 수가 작아 의미 해석은 제한적이지만, `IBTrACS + OISST` 실데이터 경로가 끝까지 동작함은 확인했다.
