import re
from minisweagent.agents.default import (
    DefaultAgent, 
    NonTerminatingException, 
    FormatError,
    ExecutionTimeoutError,
    TerminatingException, 
    Submitted,
    LimitsExceeded, 
)
from minisweagent import Environment, Model
from pydantic import BaseModel

class AgentFlyConfig(BaseModel):
    # Check the config files in minisweagent/config for example settings
    system_template: str
    instance_template: str
    timeout_template: str
    format_error_template: str
    action_observation_template: str
    action_regex: str = r"```bash\s*\n(.*?)\n```"
    step_limit: int = 0
    max_turns: int = 0
    max_model_context_length: int = 0
    cost_limit: float = 3.0

class AsyncEngineAgent(DefaultAgent):
    """Simple wrapper around DefaultAgent that uses Async Verl Agent."""
    def __init__(self, model: Model, env: Environment, tools, llm_engine, *, config_class: type = AgentFlyConfig, **kwargs):
        super().__init__(
            model=model,
            env=env,
            config_class=config_class,
            **kwargs,
        )
        self.tools = tools
        self.llm_engine = llm_engine
        self.count = 0

    async def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Run step() until agent is finished. Return exit status & message"""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.count = 0
        self.messages = []
        self.add_message("system", self.render_template(self.config.system_template))
        self.add_message("user", self.render_template(self.config.instance_template))

        while True:
            try:
                await self.step()        
            except NonTerminatingException as e:
                self.add_message("user", str(e))
            except TerminatingException as e:
                self.add_message("user", str(e))
                return type(e).__name__, str(e)

    def render_template(self, template: str, **kwargs) -> str:
        template_vars = self.config.model_dump() | self.env.get_template_vars() | self.model.get_template_vars()
        return Template(template, undefined=StrictUndefined).render(
            **kwargs, **template_vars, **self.extra_template_vars
        )

    async def step(self) -> dict:
        """Query the LM, execute the action, return the observation."""
        return self.get_observation(await self.query())

    async def query(self) -> dict:
        """Query the model and return the response."""
        self.count += 1
        
        # Check turn limits
        if 0 < self.config.step_limit < self.count:
            raise LimitsExceeded(f"Step limit reached: {self.count - 1}")

        # Check tokenized context length limit
        if self.config.max_model_context_length > 0:
            token_count = self.llm_engine.count_tokens(self.messages, tools=self.tools)
            if token_count > self.config.max_model_context_length:
                raise LimitsExceeded(f"Max model context length reached: {token_count} tokens")
        
        response = await self.llm_engine.generate_async([self.messages], tools=self.tools)
        self.add_message("assistant", response[0])
        return response[0]

    def get_observation(self, response: dict) -> dict:
        """Execute the action and return the observation."""
        output = self.execute_action(self.parse_action(response))
        observation = self.render_template(self.config.action_observation_template, output=output)
        self.add_message("user", observation)
        return output

    def parse_action(self, response: dict) -> dict:
        """Parse the action from the message. Returns the action."""
        actions = re.findall(self.config.action_regex, response, re.DOTALL)
        if len(actions) == 1:
            return {"action": actions[0].strip(), "content": response}

        raise FormatError(self.render_template(self.config.format_error_template, actions=actions))

    def execute_action(self, action: dict) -> dict:
        try:
            output = self.env.execute(action["action"])
        except (TimeoutError, subprocess.TimeoutExpired) as e:
            output = e.output.decode("utf-8", errors="replace") if getattr(e, "output", None) else ""
            raise ExecutionTimeoutError(
                self.render_template(self.config.timeout_template, action=action, output=output)
            )
        self.has_finished(output)
        return output | {"action": action["action"]}

    def has_finished(self, output: dict[str, str]):
        """Raises Submitted exception with final output if the agent has finished its task."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() in ["MINI_SWE_AGENT_FINAL_OUTPUT", "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]:
            raise Submitted("".join(lines[1:]))