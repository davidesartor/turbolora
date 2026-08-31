"""GRPO training of LoRA-XS."""

from turbolora import grpo
from turbolora.adapters import LoRAXS

parser = grpo.argument_parser()
parser.add_argument(
    "--rank", type=int, default=2, help="frozen truncated-SVD rank; R is rank x rank"
)
args = parser.parse_args()
# R's r² entries each move ~lr/step: ‖ΔR‖ ≈ lr·r
args.lr = args.lr or grpo.R_STEP_NORM / args.rank
grpo.run(args, LoRAXS, rank=args.rank)
