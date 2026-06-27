import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute worst-case robust accuracy over transfer and BPDA attacks"
    )

    parser.add_argument("--transfer_path", type=str, required=True)
    parser.add_argument("--bpda_path", type=str, required=True)

    return parser.parse_args()


def main():
    args = parse_args()

    transfer = torch.load(args.transfer_path, map_location="cpu")
    bpda = torch.load(args.bpda_path, map_location="cpu")

    labels_transfer = transfer["labels"]
    labels_bpda = bpda["labels"]

    if not torch.equal(labels_transfer, labels_bpda):
        raise ValueError("Labels do not match between transfer and BPDA files.")

    indices_transfer = transfer.get("subset_indices", None)
    indices_bpda = bpda.get("subset_indices", None)

    if indices_transfer is not None and indices_bpda is not None:
        if not torch.equal(indices_transfer, indices_bpda):
            raise ValueError("Subset indices do not match between transfer and BPDA files.")

    transfer_correct = transfer["adv_correct"].bool()
    bpda_correct = bpda["adv_correct"].bool()

    worst_case_correct = torch.logical_and(
        transfer_correct,
        bpda_correct,
    )

    n = labels_transfer.numel()

    transfer_acc = 100.0 * transfer_correct.float().mean().item()
    bpda_acc = 100.0 * bpda_correct.float().mean().item()
    worst_case_acc = 100.0 * worst_case_correct.float().mean().item()

    print("\n==============================")
    print("Worst-case robust evaluation")
    print("==============================")
    print(f"Transfer APGD-T/DLR accuracy:   {transfer_acc:.2f}%")
    print(f"BPDA APGD-T/DLR accuracy:       {bpda_acc:.2f}%")
    print("------------------------------")
    print(f"Worst-case robust accuracy:     {worst_case_acc:.2f}%")
    print(f"Robust samples:                 {worst_case_correct.sum().item()}/{n}")
    print("==============================\n")


if __name__ == "__main__":
    main()