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
GGUF_VISUALIZERS_PATH = os.path.join(PROJECT_ROOT, "gguf_visualizers")

from gguf_visualizers.image_diff_heatmapper_mk2 import ImageDiffHeatMapperMk2
from gguf_visualizers.gguf_tensor_to_image import GgufTensorToImage


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

def get_files_in_directory(path: str):
    return [
        file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))
    ]

def get_folders_in_directory(path: str):
    return [
        file for file in os.listdir(path) if not os.path.isfile(os.path.join(path, file))
    ]

def _choose_gguf_tool() -> str:
    """
    Prompts the user to select a GGUF tool from available modules.

    This function lists all available GGUF tools from both GGUF_TOOLS_PATH and
    GGUF_VISUALIZERS_PATH. It then prompts the user to select a tool either by
    entering its name or its corresponding number in the list.

    Returns:
        str: The name of the selected GGUF tool.

    Raises:
        ValueError: If an unexpected error occurs during tool selection.

    Note:
        - The function will continue prompting until a valid selection is made.
        - Tools are sorted alphabetically and numbered for easy selection.
    """

    available_modules = []
    for path in [GGUF_TOOLS_PATH, GGUF_VISUALIZERS_PATH]:
        _available_modules = sorted(
            tool for tool in os.listdir(path)
            if os.path.isfile(os.path.join(path, tool))
        )
        available_modules.extend(_available_modules)

    tool_list = "\n".join(f"{i}. {tool}" for i, tool in enumerate(available_modules, start=1))

    logger.info(f"Available gguf_tools:\n{tool_list}",f=True)

    # print(f"\nAvailable gguf_tools:\n{tool_list}")

    while True:
        gguf_tool = input("\nEnter the number or name of the gguf_tool you want to use: ").strip()
        logger.debug(f"User input: {gguf_tool}")

        if not gguf_tool:
            print("Invalid input: Please enter a non-empty gguf_tool name or number.")
            continue

        if gguf_tool.isdigit():
            index = int(gguf_tool) - 1
            if 0 <= index < len(available_modules):
                tool = available_modules[index]
                logger.debug(f"Valid input: {gguf_tool}\n Loading {tool}...", f=True)
                return tool
            else:
                print(f"Invalid number. Please enter a number between 1 and {len(available_modules)}.")
        else:
            if gguf_tool in available_modules:
                logger.debug(f"Valid input: {gguf_tool}\n Loading {gguf_tool}...", f=True)
                return gguf_tool
            else:
                print("Invalid tool name. Please enter a valid tool name or number.")

    # This line should never be reached due to the while True loop
    raise ValueError("Unexpected error in tool selection.")


def _choose_gguf_tools_arguments(gguf_tool: str) -> dict:

    gguf_tool_name = gguf_tool.__str__()
    try:
        gguf_tool = importlib.import_module(f"gguf_tools.{gguf_tool_name}")
    except ImportError:
        try:
            gguf_tool = importlib.import_module(f"gguf_visualizers.{gguf_tool_name}")
        except ImportError as e:
            logger.error(f"Failed to import module: {e}")
            return None

    logger.debug(f"gguf_tool: {gguf_tool}\ntype: {type(gguf_tool)}",f=True)
    parser: argparse.ArgumentParser = gguf_tool.create_parser()

    # Create a list of the helpers arguments.
    parser_help_as_list = _get_parser_help_as_list(parser)
    arg_list = "\n".join(f"- {arg}" for arg in parser_help_as_list)

    print(f"Name: {gguf_tool_name}")
    print(f"Description: {parser.description}")
    print(f"\nHelp list for {gguf_tool}\n{arg_list}")

    selected_args = input("""
        WARNING: YAML file constants take precedence over command line arguments
        Enter the arguments you want to run the tool with (comma-separated): 
    """)
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

    return {arg: '' for arg in chosen_args}


async def main():

    logger.info("Begin __main__")

    gguf_tool = _choose_gguf_tool()
    logger.debug(f"gguf_tool: {gguf_tool}")
    #kwargs = {} # _choose_gguf_tools_arguments(gguf_tool)

    if "image_diff_heatmapper_mk2" in gguf_tool:
        logger.info("Loading image_diff_heatmapper_mk2...")
        run = ImageDiffHeatMapperMk2()
        run.image_diff_heatmapper_mk2()

    if "gguf_tensor_to_image" in gguf_tool:
        logger.info("Loading gguf_tensor_to_image...")
        run = GgufTensorToImage()
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


