import sys

from miplib.ui import supertomo_options


def main():
    options = supertomo_options.get_register_script_options(sys.argv[1:])


if __name__ == "__main__":
    main()
