from transformers import AutoConfig, AutoModel, AutoModelForTokenClassification, \
    AutoModelForQuestionAnswering, AutoModelForSequenceClassification, AutoTokenizer
from transformers.convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS, RobertaConverter

from layoutlmft.models.layoutlmv3.configuration_layoutlmv3 import LayoutLMv3ConfigOB
from layoutlmft.models.layoutlmv3.modeling_layoutlmv3 import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3ForQuestionAnswering,
    LayoutLMv3ForSequenceClassification,
    LayoutLMv3Model,
)
from layoutlmft.models.layoutlmv3.tokenization_layoutlmv3 import LayoutLMv3TokenizerOB
from layoutlmft.models.layoutlmv3.tokenization_layoutlmv3_fast import LayoutLMv3TokenizerFastOB


AutoConfig.register("layoutlmv3OB", LayoutLMv3ConfigOB)
AutoModel.register(LayoutLMv3ConfigOB, LayoutLMv3Model)
AutoModelForTokenClassification.register(LayoutLMv3ConfigOB, LayoutLMv3ForTokenClassification)
AutoModelForQuestionAnswering.register(LayoutLMv3ConfigOB, LayoutLMv3ForQuestionAnswering)
AutoModelForSequenceClassification.register(LayoutLMv3ConfigOB, LayoutLMv3ForSequenceClassification)
AutoTokenizer.register(
    LayoutLMv3ConfigOB, slow_tokenizer_class=LayoutLMv3TokenizerOB, fast_tokenizer_class=LayoutLMv3TokenizerFastOB
)
SLOW_TO_FAST_CONVERTERS.update({"LayoutLMv3Tokenizer": RobertaConverter})
