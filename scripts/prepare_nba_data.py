import argparse
import os
import shutil


def candidate_roots(project_root, source_root):
    roots = []
    if source_root:
        roots.append(source_root)
    roots.extend([
        os.path.join(project_root, "data", "nba"),
        os.path.join(project_root, "external", "MART", "datasets", "nba"),
        os.path.join(project_root, os.pardir, "MART", "datasets", "nba"),
        os.path.join(project_root, os.pardir, "MART", "datasets", "nba_data"),
        os.path.join(project_root, os.pardir, "LED", "datasets", "nba"),
    ])
    normalized = []
    for root in roots:
        abs_root = os.path.abspath(root)
        if abs_root not in normalized:
            normalized.append(abs_root)
    return normalized


def find_pair(project_root, source_root):
    for root in candidate_roots(project_root, source_root):
        train_file = os.path.join(root, "nba_train.npy")
        test_file = os.path.join(root, "nba_test.npy")
        if os.path.isfile(train_file) and os.path.isfile(test_file):
            return root, train_file, test_file
    return None, None, None


def main():
    parser = argparse.ArgumentParser(description="Find or copy MART-style NBA npy files.")
    parser.add_argument("--source_root", default=None, help="Folder containing nba_train.npy and nba_test.npy.")
    parser.add_argument("--dest_root", default=os.path.join("data", "nba"))
    parser.add_argument("--copy", action="store_true", help="Copy found files into --dest_root.")
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    root, train_file, test_file = find_pair(project_root, args.source_root)

    if root is None:
        print("Could not find nba_train.npy and nba_test.npy.")
        print("Checked:")
        for candidate in candidate_roots(project_root, args.source_root):
            print("  - {}".format(candidate))
        print()
        print("Put MART NBA files in STAR/data/nba/ or pass --source_root <folder>.")
        return 1

    print("Found NBA data:")
    print("  train: {}".format(train_file))
    print("  test:  {}".format(test_file))

    if args.copy:
        dest_root = os.path.abspath(os.path.join(project_root, args.dest_root))
        os.makedirs(dest_root, exist_ok=True)
        shutil.copy2(train_file, os.path.join(dest_root, "nba_train.npy"))
        shutil.copy2(test_file, os.path.join(dest_root, "nba_test.npy"))
        print("Copied to: {}".format(dest_root))
    else:
        print("Use this command option:")
        print("  --nba_train_file {} --nba_test_file {}".format(train_file, test_file))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
