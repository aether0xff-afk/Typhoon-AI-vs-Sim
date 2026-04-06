# Typhoon AI vs Sim

단순 물리 모델이 예측한 태풍 강도를 인공지능이 잔차(residual) 방식으로 보정하는 실험 프로젝트다.

핵심 구조는 다음과 같다.

1. 해수면 온도(`SST`)와 위도에 따라 태풍 강도가 변하는 합성 시계열 데이터를 만든다.
2. 단순 물리 모델이 다음 시점 강도를 먼저 예측한다.
3. `MLP`, `LSTM`, `Transformer`가 물리 모델의 오차를 학습한다.
4. `물리 예측값 + AI 보정값`으로 최종 강도를 계산한다.
5. `MAE`, `RMSE`로 물리 단독 모델과 AI 보정 모델을 비교한다.

이 구조는 다음 결론에 맞춰 설계되어 있다.

> 단순 물리 모델은 태풍 강도 변화의 기본 경향을 설명할 수 있었고, 인공지능 보정 모델을 결합했을 때 물리 모델의 예측 오차를 줄여 더 정확하게 태풍 강도 변화를 예측할 수 있었다.

## 설치

```bash
python -m pip install -r requirements.txt
```

## 실행

빠른 스모크 실행:

```bash
python main.py --quick --output-dir outputs/smoke
```

기본 검증 실행:

```bash
python main.py --output-dir outputs/run
```

직접 크기 조절:

```bash
python main.py --storms 150 --epochs 50 --window-size 6 --output-dir outputs/custom
```

## 생성되는 결과물

실행이 끝나면 `output-dir` 아래에 다음 파일들이 저장된다.

- `synthetic_storm_tracks.csv`: 합성 태풍 시계열 원본 데이터
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
  experiment.py
  models.py
  plotting.py
  simulation.py
  train.py
  utils.py
```

## 구현 포인트

- 물리 모델은 `SST`, 위도, 강도 포화 효과만 반영하는 단순한 규칙 기반 모델이다.
- 실제 기준 데이터는 여기에 더해 급강화 구간, 습도, 전단, 기억 효과 같은 비선형 요소를 포함한다.
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
