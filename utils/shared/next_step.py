import re

from logger.logger import Logger
logger = Logger(logger_name=__name__,log_level=20)


def next_step(message: str, step: int=None, stop: bool=False):

    step_pattern = re.compile(r'^Step \d+', flags=re.IGNORECASE)
    match = re.match(step_pattern, message)
    asterisks = '*' * len(message)

    if stop:
        if match:
            step = int(re.search(r'\d+', match.group()).group())
        if match or step:
            current_step = step - 1
            result = input(f"{asterisks}\n{message}\n{asterisks}\nContinue to Step {step}? y/n: ")
            if result != "y":
                raise KeyboardInterrupt(f"scrape_the_law program stopped at Step {current_step}.")
            else:
                logger.info(message, f=True)
                return
        else:
            result = input(f"Continue next step? y/n: ")
            if result != "y":
                raise KeyboardInterrupt(f"scrape_the_law program stopped at step.")
            else:
                logger.info(message, f=True)
                return
    else:
        logger.info(message, f=True)
        return
