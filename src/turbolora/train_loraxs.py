"""GRPO training of LoRA-XS."""

from turbolora import grpo
from turbolora.adapters import LoRAXS

parser = grpo.argument_parser()
parser.add_argument("--rank", type=int, default=2, help="frozen truncated-SVD rank; R is rank x rank")
args = parser.parse_args()
grpo.run(args, LoRAXS, rank=args.rank)
