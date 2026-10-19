"""Production policy selected by the paired V4 rerank/filter/gate experiment."""

from edu_rag.gating import GatedWorkflow
from edu_rag.workflow import WorkflowRequest

DEFAULT_CONTEXT_POLICY = "llm_rerank_filter"


def ask(request: WorkflowRequest):
    return GatedWorkflow(context_policy=DEFAULT_CONTEXT_POLICY).run(request)


def search(request: WorkflowRequest):
    return GatedWorkflow(context_policy=DEFAULT_CONTEXT_POLICY).run(request, preview=True)
