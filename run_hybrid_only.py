from BatchAgent.test_strategies_evaluation import evaluate, STRATEGIES, SCENARIOS
import BatchAgent.test_strategies_evaluation as test_eval
test_eval.STRATEGIES = ['hybrid']
test_eval.SCENARIOS = [s for s in SCENARIOS if s['name'] in ['write code', 'modify code']]
import asyncio
asyncio.run(evaluate())
