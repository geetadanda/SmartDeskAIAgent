"SmartHR Solutions, a 500-employee SaaS company in San Ramon, CA.”

These are the 4 policy categories and the sub topics under each category.
You can query any of the following under the support chat
1.IT Support:
Password reset process
VPN configuration
Software installation requests
MFA setup
Guest Wi-Fi access
Email configuration on mobile

2.HR Policies:
Leave policy (casual, sick, earned)
WFH guidelines
Reimbursement process
Code of conduct
Anti-harassment policy
Performance review cycle

3.Onboarding:
Day-one checklist
IT equipment allocation
Benefits enrollment
Buddy programme
Probation period

4.Payroll & Comp:
Pay-day schedule
Tax declaration process
Salary structure
Stock-option vesting

"You are an HR manager at SmartHR Solutions, a 500-employee SaaS
company. Write a detailed Leave Policy document covering: casual leave
(12 days/year), sick leave (8 days/year), earned leave (15 days/year),
maternity leave (26 weeks), paternity leave (2 weeks), and compensatory
off. Include eligibility, approval process, carry-forward rules, and
encashment rules. Write in a professional tone for an employee handbook."

## Run the SmartHR Support UI

From the `SmartDeskAIAgent` directory, install the project requirements and start
the Streamlit app:

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

```bash
python -m pip install -r requirements.txt
streamlit run SmartDesk/smartHR_app.py
```

## What you can do

- **Support Chat:** Ask in-scope questions about IT support, HR, payroll and
  compensation, onboarding, company policies, the directory, and holidays.
  Answers use the available company sources. For example, ask about VPN setup,
  leave eligibility, or an upcoming holiday. Out-of-scope questions do not
  trigger a Jira ticket offer.
- **Log Jira Issue:** Submit a support issue with your work email, summary,
  description, category, and priority. Review the details and submit the form
  to create the issue in the configured Jira project.
- **Query Jira Issues:** Check an issue's summary and status by key (for example,
  `HELP-123`), or list issues associated with a requester work email.
- **Support Chat Jira follow-up:** If an in-scope support request cannot be
  answered from the available sources, prepare and review a Jira issue before
  submitting it.
- **Company Directory:** Search fictional sample employees by name, employee
  ID, department, title, email, or phone; browse the org chart to see reporting
  relationships.
- **Holiday Calendar:** Browse the listed U.S. federal holidays and observed
  dates by year, or ask Support Chat for a holiday by name or the next
  upcoming holiday.
- **Policy Documents:** Browse the available HR, IT, onboarding, and payroll
  policy documents, or ask Support Chat a question about their procedures.
- **Employment Verification:** Signed-in users can generate clearly marked
  sample letters from fictional data. Admins can also prepare official
  employment verification letters using HR-verified details.
- **Manage Users:** Admins can create accounts, reset passwords, and enable or
  disable users.

Jira features require the Jira settings described below. The directory and
sample employment records are fictional; the holiday calendar is a reference,
not a company-specific paid-holiday schedule.

See the [SmartHR high-level design](./ARCHITECTURE.svg) and its editable
[Graphviz source](./ARCHITECTURE.dot) for the main application components,
data stores, and external integrations.
The [Hybrid RAG diagram](./HYBRID_RAG.svg) shows how the support chat combines
semantic and keyword retrieval to find grounded answers.

## Evaluate answer quality

An LLM judge scores each answer from 1 to 5 on faithfulness, relevance,
completeness, and structural correctness. Test questions and the points each
answer must cover are in `evals/golden_cases.jsonl`. Keep that file out of
`data/`, because files there are added to the knowledge base.

```bash
python SmartDesk/smartHR_eval.py
```

Add `--limit 3` for a quick check or `--threshold 4.5` to change the minimum
mean score per metric. The run exits with an error if a metric falls below the
threshold or a case fails. The default judge model is `gpt-4o`; set
`OPENAI_JUDGE_MODEL` to change it. Jira tools are replaced with stubs during
evaluation, so no real issues are read or created. Full reports are saved in
`evals/results/`, which git ignores.

For DeepEval metrics, run the separate evaluator:

```bash
python SmartDesk/smartHR_deepeval.py --limit 3
```

It scores faithfulness, answer relevance, completeness against each case's
`key_points`, answer structure, and retrieval contextual precision and recall.
It also reports the harmonic-mean F1 derived from contextual precision and
recall; cases without reference key points cannot be scored for contextual
recall or F1. Use `--threshold` to set the minimum metric score (0 to 1),
`--model` to select the judge model, and `--cases` to choose a JSONL case file.
The report is saved in `evals/results/`. Jira tools are stubbed so evaluation
cannot query or create live Jira issues.

## User login

The app stores user accounts in `data/smarthr_auth.sqlite3`, using Argon2id
password hashes. On first startup, set `SMART_HR_ADMIN_USERNAME` and
`SMART_HR_ADMIN_PASSWORD` in the environment or local `.env` file to create the
initial admin account. Use a unique password with at least 12 characters.
These values are used only while the account database has no users; remove the
bootstrap password from the environment after the admin account is created.
For Streamlit Community Cloud, add both values in the app's **Settings >
Secrets** as TOML entries, for example:

```toml
SMART_HR_ADMIN_USERNAME = "hradmin"
SMART_HR_ADMIN_PASSWORD = "use-a-unique-secret-of-at-least-12-characters"
```

Keep the bootstrap password private. The app reads Streamlit secrets directly
and uses them only to create the initial admin when the account database is
empty.

Sign in with the initial admin account and use **Manage Users** in the sidebar
to create user/admin accounts, reset passwords, and disable or re-enable
accounts. Usernames are 3-32 characters. Passwords must be at least 12
characters. The first admin must be bootstrapped before the login screen can be
used. Keep the account database private and deploy the app only over HTTPS.

Set `OPENAI_API_KEY` in your environment or in the local `.env` file. On the
first support-chat request, the app loads every JSON file directly inside
`data/`, normalizes ticket records (`subject`/`description`) and FAQ records
(`question`/`answer`) into a common document shape, and builds one shared
vector and keyword index. It also reads JSONL chat data with `messages` arrays,
converting each row's last user and assistant messages to a question/answer
record. Add knowledge `.json` or `.jsonl` files there. Evaluation/configuration
JSON files outside `data/` are not indexed.
Never commit `.env` or enter passwords, MFA codes, or other sensitive
information in the chat. If the assistant cannot answer from the ticket
dataset, it classifies the request against SmartHR's employee-support scope
before searching. Out-of-scope requests (for example, general-knowledge
questions) do not invoke support tools or show a Jira ticket offer. When an
in-scope request remains unresolved, Support Chat can collect the employee's
email, summary, description, category, and priority and submit a Jira issue
after the employee confirms. Configure
`JIRA_BASE_URL`, `JIRA_EMAIL`, `JIRA_API_TOKEN`, `JIRA_PROJECT_KEY`, and
`JIRA_ISSUE_TYPE` in the local `.env` file. `JIRA_PRIORITY_MAP` can optionally
map helpdesk priorities to names configured in your Jira project. The Jira
account must have permission to create issues in that project. Users can also
ask the assistant to list their Jira tickets by providing the same work email
used when submitting them; the configured Jira account must have permission to
view issues in the project. In **Support Chat**, users can check an individual
Jira issue by key (for example `HELP-123`) or list issues by requester email.
The app sidebar provides separate **Log Jira Issue** and **Query Jira Issues**
pages, **Company Directory**, **Holiday Calendar**, **Policy Documents**, and
**Support Chat**. All signed-in users have an **Employment Verification** test
page for generating sample letters using 100 fictional records in
`data/employment_verification_testdata.csv`, including fictional annual base
salaries. Regular users see no employee
selector; their username must match the record's `account_username` field
(for example, `employee0001`). If there is no unique match, the app does not
generate a letter. Admins can select any demo employee. These fictional sample
letters are prominently marked invalid and are not stored in the account
database. For official letters, admins can use **HR Employment Letter** with
HR-verified details; it does not use the fictional directory.
Policy Documents displays the Markdown files in
`data/policies/`. The holiday page reads the U.S. federal holidays and weekday
observances for 2026 and 2027 from `data/holiday_calendar.csv`; it is a
reference list, not a company-specific paid-holiday schedule. The directory uses
fictional sample employees from `data/company_directory.csv`, including reserved
fictional phone numbers and manager IDs displayed in its **Org Chart** tab.
Replace this file with approved company directory data before using it as an
actual employee directory. Never commit the `.env` file or share the API token.
