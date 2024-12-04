import argparse
import asyncio
from io import StringIO
import importlib
import os
import sys

from config.config import PROJECT_ROOT
from logger.logger import Logger
logger = Logger(logger_name=__name__)


GGUF_TOOLS_PATH = os.path.join(PROJECT_ROOT, "gguf_tools")

from gguf_visualizers.image_diff_heatmapper_mk2 import ImageDiffHeatMapperMk2, image_diff_heatmapper_mk2_comfy_ui_node
from gguf_visualizers.gguf_tensor_to_image import GgufTensorToImage, gguf_tensor_to_image_comfy_ui_node


def _get_parser_help_as_list(parser: argparse.ArgumentParser) -> list[str]:
    """
    Get an argparser's help page and turn it into a list.
    Example:
        Use this function instead of parser.print_help()
        help_list = _get_parser_help_as_list(parser)
        print(help_list)
    """
    # Redirect stdout to a StringIO object
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    # Call print_help() which will write to our StringIO object
    parser.print_help()

    # Get the string value and restore normal stdout
    help_string = sys.stdout.getvalue()
    sys.stdout = old_stdout

    # Split the help string into a list of lines
    help_list = help_string.strip().split('\n')
    logger.debug(f"help_list: {help_list}")

    return help_list


def _choose_gguf_tool() -> str:

    available_modules = sorted(
        tool for tool in os.listdir(GGUF_TOOLS_PATH)
    )
    separator = "*" * 20
    tool_list = "\n".join(f"- {tool}" for tool in available_modules)
    print(f"{separator}\nAvailable gguf_tools:\n{tool_list}\n{separator}")

    gguf_tool = input("\nEnter the gguf_tool you want to use: ")
    gguf_tool = gguf_tool.strip()

    # If there are only spaces in gguf_tool, raise a value error.
    if not gguf_tool:
        raise ValueError("Invalid input: Please enter a non-empty gguf_tool name.")

    # List the argparse arguments for the GGUF tool
    if gguf_tool not in tool_list:
        raise ValueError("Invalid tool name.")
    else:
        return gguf_tool


def _choose_gguf_tools_arguments(gguf_tool: str) -> None:

    gguf_tool_name = gguf_tool.__name__
    gguf_tool = importlib.import_module(f"gguf_tools.{gguf_tool}.main")
    parser: argparse.ArgumentParser = gguf_tool.create_parser()

    # Create a list of the helpers arguments.
    parser_help_as_list = _get_parser_help_as_list()
    arg_list = "\n".join(f"- {arg}" for arg in parser_help_as_list)

    print(f"Name: {gguf_tool_name}")
    print(f"Description: {parser.description}")
    print(f"\nArguments for {gguf_tool}\n{arg_list}")

    selected_args = input("\nEnter the arguments you want to use: ")
    # Get and validate user input
    while True:
        selected_args = input("\nEnter the arguments you want to run the tool with (comma-separated): ").strip()
        if not selected_args:
            print("No arguments selected. Please try again.")
            continue

        chosen_args = [arg.strip().lower() for arg in selected_args.split(',')]
        logger.debug(f"chosen_args\n{chosen_args}",f=True)
        invalid_modules = [arg for arg in chosen_args if arg not in arg_list]

        if invalid_modules:
            print(f"Invalid arg(s): {', '.join(invalid_modules)}\nPlease try again.")
            continue
        
        break


async def main():

    logger.info("Begin __main__")

    gguf_tool = _choose_gguf_tool()
    kwargs = _choose_gguf_tools_arguments(gguf_tool)

    if gguf_tool == "image_diff_heatmapper_mk2":
        run = ImageDiffHeatMapperMk2(args=None, **kwargs)
        run.image_diff_heatmapper_mk2()

    if gguf_tool == "image_diff_heatmapper_mk2":
        run = GgufTensorToImage(args=None, **kwargs)
        run.gguf_tensor_to_image()

    logger.info("End __main__")

    sys.exit(0)


if __name__ == "__main__":
    import os
    base_name = os.path.basename(__file__) 
    program_name = os.path.split(os.path.split(__file__)[0])[1] if base_name != "main.py" else os.path.splitext(base_name)[0] 
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"'{program_name}' program stopped.")


