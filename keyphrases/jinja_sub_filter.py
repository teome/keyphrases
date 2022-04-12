import re
from jinja2 import pass_eval_context
from markupsafe import Markup, escape


# @pass_eval_context
# def regex_sub(eval_ctx, value, pattern, replace):
#     """Regex substitution filter

#     Example pattern and replace for css style on a word

#     pattern = r'(.*\s)(foo)(\W*)'
#     replace = r'\1<span class="keyphrase">\2</span>\3'

#     'everything is foo bar' ->
#     'everything is <span class="keyphrase">foo</span>'
#     """
#     if eval_ctx.autoescape:
#         value = escape(value)
#         # replace = Markup(replace)

#     result = re.sub(pattern, replace, value, flags=re.IGNORECASE)
#     return Markup(result) if eval_ctx.autoescape else result


@pass_eval_context
def regex_sub(eval_ctx, value, pattern, css_class):
    """Regex substitution filter

    Example pattern and replace for css style on a word

    'everything is foo bar' ->
    'everything is <span class="keyphrase">foo</span>'
    """
    # if eval_ctx.autoescape:
    # value = escape(value)
    # value = re.escape(value)

    pattern = r"(.*\s)({})(\W*)".format(pattern)
    replace = r'\1<span class="{}">\2</span>\3'.format(css_class)

    # if eval_ctx.autoescape:
    # replace = Markup(replace)

    result = re.sub(pattern, replace, value, flags=re.IGNORECASE)
    return Markup(result) if eval_ctx.autoescape else result
