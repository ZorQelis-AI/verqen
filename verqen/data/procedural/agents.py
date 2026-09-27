"""Agentic decision families: browser, tool routing, policy, triage, security, scheduling."""

from __future__ import annotations

import random

from .common import FIRST_NAMES, ITEMS, choice_q, example, noul_q, pick

SECTIONS = ["Invoices", "Orders", "Reports", "Settings", "Users", "Messages", "Billing", "Analytics", "Projects",
            "Inventory", "Contracts", "Tickets", "Payroll", "Campaigns", "Documents"]
VERBS = ["Download", "Delete", "Archive", "Share", "Export", "Approve", "Duplicate", "Rename", "Print", "Flag"]
ITEM_KIND = {"Invoices": "invoice", "Orders": "order", "Reports": "report", "Settings": "profile",
             "Users": "user", "Messages": "thread", "Billing": "statement", "Analytics": "dashboard",
             "Projects": "project", "Inventory": "SKU", "Contracts": "contract", "Tickets": "ticket",
             "Payroll": "payslip", "Campaigns": "campaign", "Documents": "file"}


def browser_task(rng: random.Random, variant: int):
    """Web navigation: home -> section -> item -> action, with login walls and cookie banners."""
    sections = pick(rng, SECTIONS, rng.randint(4, 7))
    target_sec = pick(rng, sections)
    kind = ITEM_KIND[target_sec]
    items = [f"{kind} #{rng.randint(1000, 9999)}" for _ in range(rng.randint(3, 6))]
    target_item = pick(rng, items)
    verbs = pick(rng, VERBS, rng.randint(3, 5))
    verb = pick(rng, verbs)
    goal = f"{verb} {target_item} from {target_sec}"
    page = pick(rng, ["home", "section", "wrong_section", "item", "wrong_item"])
    logged_in = rng.random() > 0.15
    banner = rng.random() < 0.15

    if page == "home":
        elements = [f"link '{s}'" for s in sections] + ["input 'Search'", "button 'Help'"]
        correct = f"click link '{target_sec}'"
        url = "/home"
    elif page == "wrong_section":
        other = pick(rng, [s for s in sections if s != target_sec])
        elements = [f"link '{ITEM_KIND[other]} #{rng.randint(1000, 9999)}'" for _ in range(3)] + ["button 'Back'", "link 'Home'"]
        correct = rng.choice(["click button 'Back'", "click link 'Home'"])
        elements_correct = {"click button 'Back'", "click link 'Home'"}
        url = f"/{other.lower()}"
    elif page == "section":
        elements = [f"link '{it}'" for it in items] + ["button 'Back'", "input 'Filter'"]
        correct = f"click link '{target_item}'"
        url = f"/{target_sec.lower()}"
    elif page == "wrong_item":
        other = pick(rng, [i for i in items if i != target_item])
        elements = [f"button '{v}'" for v in verbs] + ["button 'Back'"]
        correct = "click button 'Back'"
        url = f"/{target_sec.lower()}/{other.split('#')[1]}"
        target_item_shown = other
    else:
        elements = [f"button '{v}'" for v in verbs] + ["button 'Back'"]
        correct = f"click button '{verb}'"
        url = f"/{target_sec.lower()}/{target_item.split('#')[1]}"

    actions = [f"click {e}" if not e.startswith("input") else f"type into {e}" for e in elements]
    good = {correct}
    if page == "wrong_section":
        good = elements_correct
    if not logged_in:
        actions = ["type into input 'Username'", "type into input 'Password'", "click button 'Sign in'", "click link 'Home'"]
        good = {"type into input 'Username'"}
        url = "/login"
        elements = ["input 'Username'", "input 'Password'", "button 'Sign in'", "link 'Home'"]
    elif banner:
        actions = ["click button 'Accept all cookies'", "click button 'Reject non-essential cookies'"] + actions[:4]
        good = {"click button 'Reject non-essential cookies'"}
        elements = ["cookie banner covering the page"] + elements
    shown = f"Page {url}. Visible elements: {', '.join(elements)}."
    if page == "wrong_item" and logged_in and not banner:
        shown += f" Header: {target_item_shown}."
    elif page == "item" and logged_in and not banner:
        shown += f" Header: {target_item}."
    state = (f"Browser agent. Task: {goal}. Privacy policy: never accept non-essential cookies. {shown}"
             if variant % 2 else f"GOAL: {goal}\nPOLICY: decline optional cookies\nBROWSER: {shown}")
    actions = list(dict.fromkeys(actions))
    return example("browser", state, [choice_q("Which action should the agent take next?", actions,
                                               [1.0 if a in good else 0.0 for a in actions], rng)], variant)


TOOLS = [
    ("get_weather", "returns the current weather and forecast for a city", "Will it rain in {city} tomorrow?"),
    ("run_sql", "runs a read-only SQL query against the company database", "How many orders did we ship last week according to our database?"),
    ("calculator", "evaluates arithmetic expressions exactly", "What is 17.5% of 2,340?"),
    ("send_email", "sends an email to a recipient", "Email {name} that the meeting moved to Friday."),
    ("calendar_create", "creates a calendar event", "Put a dentist appointment on my calendar for Tuesday 3pm."),
    ("web_search", "searches the public web for recent information", "Who won yesterday's championship final?"),
    ("translate", "translates text between languages", "Translate 'where is the station' into Spanish."),
    ("read_file", "reads the contents of a local file", "Show me what is inside notes/{item}.txt."),
    ("run_python", "executes a Python snippet in a sandbox", "Execute this script and tell me what it prints."),
    ("resize_image", "resizes or crops an image", "Make this photo 800 pixels wide."),
    ("stock_quote", "returns the latest price of a stock ticker", "What is ACME's share price right now?"),
    ("convert_currency", "converts an amount between currencies", "How much is 250 euros in yen?"),
    ("get_directions", "plans a route between two places", "How do I drive from {city} to the airport?"),
    ("spell_check", "finds spelling mistakes in text", "Fix the typos in my cover letter."),
    ("create_ticket", "opens a support ticket in the helpdesk", "Log a bug: the checkout page crashes on submit."),
    ("set_reminder", "schedules a reminder notification", "Remind me to water the plants in two hours."),
    ("summarize_doc", "summarizes a long document", "Give me the key points of this 40-page report."),
    ("order_lookup", "looks up the status of a customer order", "Where is order #{num} right now?"),
    ("refund_payment", "issues a refund for a payment", "Refund the duplicate charge on invoice #{num}."),
    ("speech_to_text", "transcribes an audio recording", "Transcribe this voicemail."),
]


def tool_routing(rng: random.Random, variant: int):
    k = rng.randint(3, 6)
    chosen = rng.sample(TOOLS, k)
    target = pick(rng, chosen)
    alias = rng.random() < 0.4  # opaque tool names force reading the descriptions
    names = {t[0]: (f"tool_{rng.randint(100, 999)}" if alias else t[0]) for t in chosen}
    request = target[2].format(city=pick(rng, ["Lyon", "Pune", "Osaka", "Denver", "Nairobi"]),
                               name=pick(rng, FIRST_NAMES), item=pick(rng, ITEMS), num=rng.randint(10000, 99999))
    listing = "\n".join(f"- {names[t[0]]}: {t[1]}" for t in chosen)
    state = f"Available tools:\n{listing}\nUser request: \"{request}\""
    opts = [names[t[0]] for t in chosen]
    qs = [choice_q("Which tool should be called?", opts, opts.index(names[target[0]]), rng)]
    return example("tool_routing", state, qs, variant)


def refund_policy(rng: random.Random, variant: int):
    want = rng.choice(["approve", "deny", "escalate to manager"])  # label-first via rejection sampling
    for _ in range(200):
        ex = _refund_case(rng, variant)
        if ex[1] == want:
            return ex[0]
    return ex[0]


def _refund_case(rng: random.Random, variant: int):
    days_limit = rng.choice([14, 30, 45, 60])
    amount_limit = rng.choice([100, 200, 500, 1000])
    days = rng.randint(1, 90)
    amount = rng.randint(10, 2000)
    opened_digital = rng.random() < 0.2
    vip = rng.random() < 0.2
    if opened_digital:
        decision = "deny"
    elif days > days_limit and not vip:
        decision = "deny"
    elif amount > amount_limit:
        decision = "escalate to manager"
    else:
        decision = "approve"
    policy = (f"Refund policy: refunds are accepted within {days_limit} days of delivery (VIP members: no time limit). "
              f"Refunds above ${amount_limit} need manager approval. Opened digital products are never refundable.")
    case = (f"Case: {pick(rng, FIRST_NAMES)} requests a ${amount} refund for a {'digital download (opened)' if opened_digital else pick(rng, ITEMS)} "
            f"delivered {days} days ago. VIP member: {'yes' if vip else 'no'}.")
    state = f"{policy}\n{case}" if variant % 2 else f"{case}\n{policy}"
    opts = ["approve", "deny", "escalate to manager"]
    qs = [choice_q("What is the correct decision under the policy?", opts, opts.index(decision), rng)]
    if rng.random() < 0.4:
        qs.append(noul_q("Is the request inside the time window?", days <= days_limit or vip))
    return example("refund_policy", state, qs, variant), decision


def incident_triage(rng: random.Random, variant: int):
    users = rng.choice([1, 5, 40, 300, 5000, 100000])
    data_loss = rng.random() < 0.2
    security = rng.random() < 0.15
    workaround = rng.random() < 0.5
    revenue = rng.random() < 0.3
    points = (2 if users >= 1000 else 1 if users >= 50 else 0) + 2 * data_loss + 2 * security + revenue - workaround
    sev = max(1, min(5, 1 + points))
    levels = ["SEV-5 (lowest)", "SEV-4", "SEV-3", "SEV-2", "SEV-1 (critical)"]
    rubric = ("Severity rubric: start at SEV-5. Escalate one level if 50+ users are affected, two levels if 1000+. "
              "Escalate two levels for data loss and two for a security impact, one for revenue impact. "
              "De-escalate one level if a workaround exists. SEV-1 is the maximum.")
    report = (f"Incident: {users} users affected; data loss: {'yes' if data_loss else 'no'}; security impact: "
              f"{'yes' if security else 'no'}; revenue impact: {'yes' if revenue else 'no'}; workaround available: "
              f"{'yes' if workaround else 'no'}.")
    state = f"{rubric}\n{report}" if variant % 2 else f"{report}\n{rubric}"
    qs = [choice_q("Assign the severity level.", levels, sev - 1, rng, qtype="score")]
    if rng.random() < 0.5:
        qs.append(noul_q("Should the on-call engineer be paged immediately (SEV-2 or higher)?", sev >= 4))
    return example("incident_triage", state, qs, variant)


BRANDS = [("PayPal", "paypal.com"), ("Amazon", "amazon.com"), ("Netflix", "netflix.com"), ("Chase", "chase.com"),
          ("Microsoft", "microsoft.com"), ("Apple", "apple.com"), ("DHL", "dhl.com"), ("Google", "google.com")]


def _lookalike(domain: str, rng: random.Random) -> str:
    name, tld = domain.split(".", 1)
    tricks = [name.replace("o", "0", 1), name.replace("l", "1", 1), name + "-secure", name + "-support",
              name.replace("a", "4", 1), "login-" + name]
    cands = [t for t in tricks if t != name]
    return f"{rng.choice(cands)}.{rng.choice([tld, 'net', 'co', 'xyz'])}"


def phishing(rng: random.Random, variant: int):
    brand, domain = pick(rng, BRANDS)
    verdict = rng.choice(["legitimate", "phishing", "suspicious, needs review"])
    sender = link = replyto = domain
    if verdict == "phishing":
        if rng.random() < 0.5:
            link = _lookalike(domain, rng)
        else:
            sender = _lookalike(domain, rng)
    elif verdict == "suspicious, needs review":
        replyto = f"{pick(rng, ['gmail.com', 'outlook.com', 'proton.me'])}"
    urgency = rng.random() < 0.5
    subject = pick(rng, ["Your account statement", "Action required: verify your details", "Delivery update",
                         "Security alert", "Your receipt"])
    state = (f"Email claiming to be from {brand}. Subject: \"{subject}\". From: no-reply@{sender}. Reply-To: help@{replyto}. "
             f"Link in body: https://{link}/account. {'Tone: urgent, asks to act within 24 hours.' if urgency else 'Tone: routine.'} "
             f"Rules: a sender or link domain that is not exactly {domain} means phishing; a mismatched Reply-To alone means "
             f"suspicious; urgency by itself is not evidence.")
    opts = ["legitimate", "phishing", "suspicious, needs review"]
    qs = [choice_q("Classify this email.", opts, opts.index(verdict), rng)]
    if rng.random() < 0.4:
        qs.append(noul_q("Is it safe to click the link?", verdict == "legitimate"))
    return example("phishing", state, qs, variant)


def scheduling(rng: random.Random, variant: int):
    people = pick(rng, FIRST_NAMES, rng.randint(2, 4))
    slots = ["Mon 9:00", "Mon 14:00", "Tue 10:00", "Tue 16:00", "Wed 11:00", "Thu 9:00", "Thu 15:00", "Fri 13:00"]
    options = pick(rng, slots, rng.randint(3, 5))
    answer = pick(rng, options)
    busy = {}
    for p in people:
        b = set(rng.sample([s for s in slots if s != answer], rng.randint(1, 4)))
        busy[p] = b
    others = [s for s in options if s != answer]
    for s in others:  # make every non-answer option infeasible for someone
        if not any(s in busy[p] for p in people):
            busy[pick(rng, people)].add(s)
    lines = "; ".join(f"{p} is busy {', '.join(sorted(busy[p]))}" for p in people)
    state = f"Find a meeting slot for {', '.join(people)}. Availability: {lines}."
    return example("scheduling", state, [choice_q("Which slot works for everyone?", options, options.index(answer), rng)], variant)


FAMILIES = {
    "browser": browser_task,
    "tool_routing": tool_routing,
    "refund_policy": refund_policy,
    "incident_triage": incident_triage,
    "phishing": phishing,
    "scheduling": scheduling,
}
