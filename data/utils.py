import os
import jinja2
import logging


class SuppressLogging:
    def __enter__(self):
        logging.Logger.manager.loggerDict = {}
        self.original_logging_method = logging.Logger._log
        logging.Logger._log = lambda *args, **kwargs: None

    def __exit__(self, exc_type, exc_value, traceback):
        logging.Logger._log = self.original_logging_method


class JinjaTemplateManager:
    ''' Loads templates from gpt3/tempalates and renders them with jinja2'''

    def __init__(self, template_dir: str = './templates'):
        self.env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir))

    def render(self, template_name: str, **kwargs) -> str:
        ''' Render a template with the given arguments'''
        template = self.env.get_template(template_name)
        rendered_text = template.render(**kwargs)
        return rendered_text
