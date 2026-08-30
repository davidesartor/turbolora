"""GRPO training of the conventional LoRA baseline."""

from turbolora import grpo
from turbolora.adapters import LoRA

parser = grpo.argument_parser()
parser.add_argument("--rank", type=int, default=32)
args = parser.parse_args()
grpo.run(args, LoRA, rank=args.rank)
