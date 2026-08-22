"""OpenUI generative-UI instructions: the `get_openui_instructions` tool.

Pattern borrowed from TrueForge's deferred-instructions design: instead of
having the agent call "apps"/MCP servers that build UI on the server, the
agent is *told* how to author UI directly and emits it as data. A short hint
lives in the system prompt; the full OpenUI authoring instruction (fencing
rules, component signatures, built-in functions, examples, rules) is only
loaded on demand through one tool call — the prompt stays small while the
model gets the exact contract the frontend renders.

When the agent wants interactive UI that markdown cannot express (dashboards,
charts, tables, forms, KPI cards), it:

1. calls `get_openui_instructions` (no arguments) to load the instructions;
2. emits a fenced ```openui code block in its answer;

The frontend renders the fence with `@openuidev/react-ui`'s `Renderer`
(`openuiLibrary`), so the component signatures below must match that library
(the generative-UI component set: Stack, Tabs, Card, charts, Table, Form,
Callout, ...). The instruction is authored in the same style as the library's
own prompt generator output so the model produces valid OpenUI Lang.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel


class GetOpenUIInstructionsInput(BaseModel):
    """No arguments — the full instruction set is returned."""


# The full OpenUI authoring instruction the agent must load before writing any
# ```openui block. Tracks the openuiLibrary shipped in @openuidev/react-ui:
# keep signatures in sync with whatever version the frontend renders with.
OPENUI_INSTRUCTIONS = """\
OpenUI generative UI — authoring instructions

FENCING
All openui code must be fenced within an openui block:
```openui
code
```

SYNTAX RULES
1. Each statement is on its own line: identifier = Expression
2. root is the entry point — every program must define root = Stack(...)
3. Expressions are: strings ("..."), numbers, booleans (true/false), null,
   arrays ([...]), objects ({...}), or component calls TypeName(arg1, arg2, ...)
4. Use references for readability: define name = ... on one line, then use
   name later
5. EVERY variable (except root) MUST be referenced by at least one other
   variable. Unreferenced variables are silently dropped and will NOT render.
   Always include defined variables in their parent's children/items array.
6. Arguments are POSITIONAL (order matters, not names). Write
   Stack([children], "row", "l") NOT Stack([children], direction: "row",
   gap: "l") — colon syntax is NOT supported and silently breaks
7. Optional arguments can be omitted from the end
8. Strings use double quotes with backslash escaping

COMPONENT SIGNATURES
Arguments marked with ? are optional. Sub-components can be inline or
referenced; prefer references for better streaming. Props typed
ActionExpression accept an Action([@steps...]) expression — available steps
are @ToAssistant(message), @OpenUrl(url), @Set($var, value),
@Reset($var1, ...). Props marked $binding<type> accept a $variable reference
for two-way binding.

Layout:
Stack([children], direction?: "row" | "column", gap?: "none" | "xs" | "s" | "m" | "l" | "xl" | "2xl", align?: "start" | "center" | "end" | "stretch" | "baseline", justify?: "start" | "center" | "end" | "between" | "around" | "evenly", wrap?: boolean) — Flex container. direction: "row"|"column" (default "column"). gap values default "m" if omitted.
Tabs(items: TabItem[]) — Tabbed container; shared $variable bindings work across all items
TabItem(value: string, trigger: string, content: (component array)) — value is unique id, trigger is tab label, content is an array of components
Accordion(items: AccordionItem[]) — Collapsible sections
AccordionItem(value: string, trigger: string, content: (component array)) — value is unique id, trigger is section title
Steps(items: StepsItem[]) — Step-by-step guide
StepsItem(title: string, details: string) — title and details text for one step
Carousel(children: (component array)[], variant?: "card" | "sunk") — Horizontal scrollable carousel
Separator(orientation?: "horizontal" | "vertical", decorative?: boolean) — Visual divider between content sections
Modal(title: string, open?: $binding<boolean>, children: (component array), size?: "sm" | "md" | "lg") — Modal dialog. open is a reactive $boolean binding — set it true to open; X/Escape/backdrop auto-close. Put a Form with its own buttons inside children.
- For grid-like layouts, use Stack with direction "row" and wrap set to true.
- Prefer justify "start" (or omit justify) with wrap=true for stable columns.
- Use nested Stacks when you need explicit rows/sections.
- Show/hide sections: $editId != "" ? Card([editForm]) : null
- Use Tabs for alternative views (chart types, data sections) — no $variable needed.

Content:
Card(children: (component array), variant?: "card" | "sunk" | "clear", direction?: "row" | "column", gap?: ..., align?: ..., justify?: ..., wrap?: boolean) — Styled container. variant: "card" (default, elevated) | "sunk" (recessed) | "clear" (transparent). Always full width. Cards flex to share space in row/wrap layouts.
CardHeader(title?: string, subtitle?: string) — Header with optional title and subtitle
TextContent(text: string, size?: "small" | "default" | "large" | "small-heavy" | "large-heavy") — Text block; supports markdown
MarkDownRenderer(textMarkdown: string, variant?: "clear" | "card" | "sunk") — Renders markdown text with optional container variant
Callout(variant: "info" | "warning" | "error" | "success" | "neutral", title: string, description: string, visible?: $binding<boolean>) — Callout banner; optional visible is reactive — auto-dismisses after 3s by setting $visible to false
TextCallout(variant?: "neutral" | "info" | "warning" | "success" | "danger", title?: string, description?: string) — Text callout with variant, title, and description
Image(alt: string, src?: string) — Image with alt text and optional URL
ImageBlock(src: string, alt?: string) — Image block with loading state
ImageGallery(images: {src: string, alt?: string, details?: string}[]) — Gallery grid of images with modal preview
CodeBlock(language: string, codeString: string) — Syntax-highlighted code block
- Use Cards to group related KPIs or sections. Stack with direction "row" for side-by-side layouts.
- Success toast: Callout("success", "Saved", "Done.", $showSuccess) — use @Set($showSuccess, true) in the save action; auto-dismisses after 3s. Errors: result.status == "error" ? Callout("error", "Failed", result.error) : null
- KPI card: Card([TextContent("Label", "small"), TextContent("" + @Count(@Filter(data.rows, "field", "==", "value")), "large-heavy")])

Tables:
Table(columns: Col[]) — Data table — column-oriented. Each Col holds its own data array.
Col(label: string, data, type?: "string" | "number" | "action") — Column definition — holds label + data array
- Table is COLUMN-oriented: Table([Col("Label", dataArray), Col("Count", countArray, "number")]). Use array pluck for data: data.rows.fieldName
- Col data can be component arrays for styled cells: Col("Status", @Each(data.rows, "item", Tag(item.status, null, "sm", item.status == "open" ? "success" : "danger")))
- Row actions: Col("Actions", @Each(data.rows, "t", Button("Edit", Action([@Set($showEdit, true), @Set($editId, t.id)]))))
- Sortable: sorted = @Sort(data.rows, $sortField, "desc"). Bind $sortField to a Select. Use sorted.fieldName for Col data
- Searchable: filtered = @Filter(data.rows, "title", "contains", $search). Bind $search to an Input
- Chain sort + filter: filtered = @Filter(...) then sorted = @Sort(filtered, ...) — use sorted for both Table and Charts
- Empty state: @Count(data.rows) > 0 ? Table([...]) : TextContent("No data yet")

Forms:
Form(name: string, buttons: Buttons, fields: FormControl[]) — The second argument is the submit/action row. Never nest Form inside Form.
FormControl(label: string, control) — Label + one input control
Input(name: string, placeholder?: string, type?: "text" | "email" | "password" | "number" | "date", options?: {required?: boolean, minLength?: number, email?: boolean, ...}) — Text/number input with validation options
TextArea(name: string, placeholder?: string, rows?: number, options?: {required?: boolean, minLength?: number, ...}) — Multiline input
Select(name: string, items: SelectItem[], placeholder?: string, options?: {required?: boolean}) — Dropdown; bind with $var for reactive selection
SelectItem(value: string, label: string) — One dropdown option
Buttons(buttons: Button[]) — A row of buttons (usually the form's submit row)
Button(label: string, action: ActionExpression, variant?: "primary" | "secondary" | "subtle" | "danger") — A clickable button; label + Action + optional variant
- Define one FormControl reference per field so controls stream progressively.
- Action([@ToAssistant("Submit")]) sends the typed field values back to the assistant — forms whose submit needs real work send the message back instead of faking success.
- After a successful flow, consider a success Callout and @Reset($var1, $var2) to restore defaults.

Charts (2D):
BarChart(labels: string[], series: Series[], variant?: "grouped" | "stacked", xLabel?: string, yLabel?: string) — Vertical bars; compare values across categories with one or more series
LineChart(labels: string[], series: Series[], variant?: "linear" | "natural" | "step", xLabel?: string, yLabel?: string) — Lines over categories; trends and continuous data over time
AreaChart(labels: string[], series: Series[], variant?: "linear" | "natural" | "step", xLabel?: string, yLabel?: string) — Filled area under lines; cumulative totals or volume trends
RadarChart(labels: string[], series: Series[]) — Spider/web chart; compare multiple variables across entities
HorizontalBarChart(labels: string[], series: Series[], variant?: "grouped" | "stacked", xLabel?: string, yLabel?: string) — Horizontal bars; prefer when category labels are long or for ranked lists
Series(category: string, values: number[]) — One data series
- Charts accept column arrays: LineChart(labels, [Series("Name", values)]). Use array pluck: LineChart(data.rows.day, [Series("Views", data.rows.views)])
- Wrap charts in Cards with CardHeader for titled sections.
- Multiple chart views: use Tabs — Tabs([TabItem("line", "Line", [LineChart(...)]), TabItem("bar", "Bar", [BarChart(...)])])

Charts (1D):
PieChart(labels: string[], values: number[], variant?: "pie" | "donut") — Circular slices; use plucked arrays: PieChart(data.categories, data.values)
RadialChart(labels: string[], values: number[]) — Radial bars; use plucked arrays
SingleStackedBarChart(labels: string[], values: number[]) — Single horizontal stacked bar; use plucked arrays
Slice(category: string, value: number) — One slice with label and numeric value
- Pie/Bar charts need NUMBERS, not objects. For list data aggregate with @Count(@Filter(...)): PieChart(["Low", "Med", "High"], [@Count(@Filter(data.rows, "priority", "==", "low")), @Count(@Filter(data.rows, "priority", "==", "medium")), @Count(@Filter(data.rows, "priority", "==", "high"))], "donut")
- KPI from count: TextContent("" + @Count(@Filter(data.rows, "status", "==", "open")), "large-heavy")

Charts (Scatter):
ScatterChart(datasets: ScatterSeries[], xLabel?: string, yLabel?: string) — X/Y scatter plot; correlations, distributions, clustering
ScatterSeries(name: string, points: Point[]) — Named dataset
Point(x: number, y: number, z?: number) — Data point with numeric coordinates

Data Display:
TagBlock(tags: string[]) — tags is an array of strings
Tag(text: string, icon?: string, size?: "sm" | "md" | "lg", variant?: "neutral" | "info" | "success" | "warning" | "danger") — Styled tag/badge
- Color-mapped Tag: Tag(value, null, "sm", value == "high" ? "danger" : value == "medium" ? "warning" : "neutral")

BUILT-IN FUNCTIONS
Data functions prefixed with @ to distinguish them from components. These are
the ONLY functions available — do NOT invent new ones. Use them on Query/data
results — do NOT hardcode computed values.

@Count(array) → number — Array length
@First(array) → element — First element
@Last(array) → element — Last element
@Sum(numbers[]) → number — Sum of numeric array
@Avg(numbers[]) → number — Average of numeric array
@Min(numbers[]) → number — Minimum value
@Max(numbers[]) → number — Maximum value
@Sort(array, field, direction?) → sorted array — direction "asc" (default) or "desc"
@Filter(array, field, operator: "==" | "!=" | ">" | "<" | ">=" | "<=" | "contains", value) → filtered array
@Round(number, decimals?) → number — Round to N decimal places (default 0)
@Abs(number) → number — Absolute value
@Floor(number) → number — Round down
@Ceil(number) → number — Round up
@Each(array, varName, template) — Evaluate template for each element. varName
  is the loop variable — use it ONLY inside the template expression (inline).
  Do NOT create a separate statement for the template.

Builtins compose — output of one is input to the next:
@Count(@Filter(data.rows, "field", "==", "val")) for KPIs/chart values,
@Round(@Avg(data.rows.score), 1), @Each(data.rows, "item", Comp(item.field))
for per-item rendering. Array pluck: data.rows.field extracts a field from
every row → use with @Sum, @Avg, charts, tables.

IMPORTANT @Each rule: the loop variable (e.g. "item") is ONLY available inside
the @Each template expression. Always inline the template — do NOT extract it
to a separate statement.
CORRECT: Col("Actions", @Each(rows, "t", Button("Edit", Action([@Set($id, t.id)]))))
WRONG: myBtn = Button("Edit", Action([@Set($id, t.id)])) then
Col("Actions", @Each(rows, "t", myBtn)) — t is undefined in myBtn.

REACTIVE VARIABLES
$variables hold state. They bind to form controls and update pages live.
@Set($var, value) — assign a value to a $variable
@Reset($var1, $var2, ...) — restore a $variable to its default
Changes re-evaluate every expression referencing the variable automatically.

HOISTING AND STREAMING
openui-lang supports hoisting: a reference can be used BEFORE it is defined;
the parser resolves all references after the full input is parsed. During
streaming the output is re-parsed on every chunk, so undefined references
appear once their definitions stream in — a progressive top-down reveal.

Recommended statement order for optimal streaming:
1. root = Stack(...) — UI shell appears immediately
2. $variable declarations — state ready for bindings
3. Query statements — defaults resolve immediately so components render with data
4. Component definitions — fill in with data already available
5. Data values — leaf content last

Always write the root = Stack(...) statement first so the UI shell appears
immediately, even before child data has streamed in.

EXAMPLES
Example 1 — Table (column-oriented):
root = Stack([title, tbl])
title = TextContent("Top Languages", "large-heavy")
tbl = Table([Col("Language", langs), Col("Users (M)", users), Col("Year", years)])
langs = ["Python", "JavaScript", "Java", "TypeScript", "Go"]
users = [15.7, 14.2, 12.1, 8.5, 5.2]
years = [1991, 1995, 1995, 2012, 2009]

Example 2 — Bar chart:
root = Stack([title, chart])
title = TextContent("Q4 Revenue", "large-heavy")
chart = BarChart(labels, [s1, s2], "grouped")
labels = ["Oct", "Nov", "Dec"]
s1 = Series("Product A", [120, 150, 180])
s2 = Series("Product B", [90, 110, 140])

Example 3 — Form with validation:
root = Stack([title, form])
title = TextContent("Contact Us", "large-heavy")
form = Form("contact", btns, [nameField, emailField, countryField, msgField])
nameField = FormControl("Name", Input("name", "Your name", "text", { required: true, minLength: 2 }))
emailField = FormControl("Email", Input("email", "you@example.com", "email", { required: true, email: true }))
countryField = FormControl("Country", Select("country", countryOpts, "Select...", { required: true }))
msgField = FormControl("Message", TextArea("message", "Tell us more...", 4, { required: true, minLength: 10 }))
countryOpts = [SelectItem("us", "United States"), SelectItem("uk", "United Kingdom"), SelectItem("de", "Germany")]
btns = Buttons([Button("Submit", Action([@ToAssistant("Submit")]), "primary"), Button("Cancel", Action([@ToAssistant("Cancel")]), "secondary")])

Example 4 — Tabs with mixed content:
root = Stack([title, tabs])
title = TextContent("React vs Vue", "large-heavy")
tabs = Tabs([tabReact, tabVue])
tabReact = TabItem("react", "React", reactContent)
tabVue = TabItem("vue", "Vue", vueContent)
reactContent = [TextContent("React is a library by Meta for building UIs."), Callout("info", "Note", "React uses JSX syntax.")]
vueContent = [TextContent("Vue is a progressive framework by Evan You."), Callout("success", "Tip", "Vue has a gentle learning curve.")]

IMPORTANT RULES
- When asked about data, generate realistic/plausible data.
- Choose components that best represent the content (tables for comparisons,
  charts for trends, forms for input, etc.).
- When you render data in an openui block (tables, charts, KPI cards), do NOT
  repeat the same numbers/facts in the markdown text outside the block.
- Text outside the openui block can optionally include qualitative insights,
  actionable next steps, and caveats/context the visuals do not show.
- If all information is already visible in the openui components, a brief
  one-line summary is sufficient — do not enumerate the same values again.

FINAL VERIFICATION
Before finishing, walk your output and verify:
1. root = Stack(...) is the FIRST line (for optimal streaming).
2. Every referenced name is defined. Every defined name (other than root) is
   reachable from root.
- For grid-like layouts, use Stack with direction "row" and wrap=true. Avoid
  justify="between" unless you specifically want large gutters.
- For forms, define one FormControl reference per field so controls can stream
  progressively; always provide the second Form argument with Buttons(...)
  actions: Form(name, buttons, fields). Never nest Form inside Form.
- Use @Reset($var1, $var2) after form submit to restore defaults — not
  @Set($var, "").
- $variables are reactive: changing via Select or @Set re-evaluates all
  expressions referencing them.
- Use existing components (Tabs, Accordion, Modal) before inventing ternary
  show/hide patterns.

USER INTERACTION CHECKLIST
1. Will the response be UI heavy and contain components markdown cannot
   express? Use openui if the answer is yes.
2. ```openui fencing must be closed.
"""


async def _get_openui_instructions() -> str:
    """Tool body: return the full OpenUI authoring instruction."""
    return OPENUI_INSTRUCTIONS


def build_openui_instructions_tool() -> BaseTool:
    """The `get_openui_instructions` tool: generative-UI authoring rules.

    Always registered (core capability, like `publish_skill`) so the agent
    can emit valid ```openui blocks without the payload living in every
    system prompt — loading it is one cheap tool call, only when the answer
    is UI-heavy.
    """
    return StructuredTool.from_function(
        coroutine=_get_openui_instructions,
        name="get_openui_instructions",
        description=(
            "Load the full OpenUI generative-UI authoring instructions "
            "(fencing rules, component signatures, built-in functions, "
            "examples, rules). Call this BEFORE emitting any ```openui "
            "fenced code block. Pass an empty object as arguments: {}."
        ),
        args_schema=GetOpenUIInstructionsInput,
    )
