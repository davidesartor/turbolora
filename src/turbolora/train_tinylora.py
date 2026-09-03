"""GRPO training of TinyLoRA."""

from turbolora import grpo
from turbolora.adapters import TinyLoRA

parser = grpo.argument_parser()
parser.add_argument("--rank", type=int, default=2, help="frozen truncated-SVD rank")
parser.add_argument(
    "--proj-dim",
    type=int,
    default=None,
    help="u: trainable vector size per group (default r², the full R basis)",
)
parser.add_argument(
    "--untie",
    action="store_true",
    help="one v per module (default: one global v shared by every module)",
)
args = parser.parse_args()
proj_dim = args.proj_dim or args.rank**2
# u params each move ~lr/step along a Pᵢ of norm r: ‖ΔR‖ ≈ lr·r·√u
args.lr = args.lr or grpo.R_STEP_NORM / (args.rank * proj_dim**0.5)
# tie=0 -> one global v; untied with u = r² is LoRA-XS with a random basis
grpo.run(args, TinyLoRA, rank=args.rank, proj_dim=proj_dim, tie=1 if args.untie else 0)
