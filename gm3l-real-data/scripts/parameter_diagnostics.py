#!/usr/bin/env python3
"""Evaluate native parameter estimates against the training-label oracle."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared",type=Path,required=True)
    parser.add_argument("--runs",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--methods",nargs="+",default=["SMLE","FINAL"])
    parser.add_argument("--plot",action="store_true")
    parser.add_argument("--synthetic",action="store_true",help="Explicitly mark validation figures as synthetic")
    args=parser.parse_args(argv)
    from gm3l_real.parameters import evaluate_parameters,plot_parameters
    rows=evaluate_parameters(args.prepared,args.runs,args.output,args.methods)
    if args.plot:plot_parameters(args.output/"parameter_errors.csv",args.output,args.methods,synthetic=args.synthetic)
    return 0 if all(row["status"]=="complete" for row in rows) else 2


if __name__=="__main__":
    raise SystemExit(main())
