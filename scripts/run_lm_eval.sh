python -m eval.run_lm_eval \
  --train-config configs/gdn_340m_opd.yaml \
  --tasks arc_easy \
  --num-fewshot 0 \
  --device cuda:5 \
  --batch-size 128


python -m eval.run_lm_eval \
  --train-config configs/gdn_340m_opd.yaml \
  --checkpoint outputs/gdn340m-fineweb-opd/checkpoints/step_00000100.pt \
  --tasks arc_challenge \
  --num-fewshot 0 \
  --device cuda:5 \
  --batch-size 8

python -m eval.run_lm_eval \
  --train-config configs/gdn_340m_opd.yaml \
  --tasks ifeval \
  --batch-size 256 \
  --device cuda:5

python -m eval.run_lm_eval \
  --train-config configs/gdn_340m_opd.yaml \
  --checkpoint outputs/gdn340m-fineweb-opd/checkpoints/step_00000100.pt \
  --tasks ifeval \
  --batch-size 256 \
  --device cuda:6
