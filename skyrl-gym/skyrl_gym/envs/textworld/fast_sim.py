"""
Fast pure-Python TextWorld game simulator.

Reads the .json game specification files directly, bypassing the slow
Inform7 subprocess engine. Implements all TextWorld commands and tracks
game state, quest progress, and generates text observations.

This is designed for RL training and data generation where speed matters.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple


class FastTextWorldSimulator:
    """
    Pure-Python TextWorld game engine.

    Loads the .json game spec and simulates the game entirely in-process.
    No subprocesses, no IPC, no global locks.

    Usage:
        sim = FastTextWorldSimulator(game_json_path)
        obs, info = sim.reset()
        obs, reward, done, info = sim.step("go north")
    """

    def __init__(self, game_json_path: str):
        with open(game_json_path, "r") as f:
            self._spec = json.load(f)

        # Build info lookup: entity_id -> info dict
        self._infos: Dict[str, Dict[str, Any]] = {}
        for eid, info in self._spec["infos"]:
            self._infos[eid] = info

        # Build name -> entity_id lookup (lowercase name to id)
        # Multiple entities can share the same noun; adj disambiguates.
        self._name_to_ids: Dict[str, List[str]] = {}
        for eid, info in self._infos.items():
            name = info.get("name")
            if name:
                key = name.lower()
                self._name_to_ids.setdefault(key, []).append(eid)
                # Also index by noun alone
                noun = info.get("noun")
                if noun and noun.lower() != key:
                    self._name_to_ids.setdefault(noun.lower(), []).append(eid)

        # Parse quest info -- games can have multiple incremental quests
        self._objective = self._spec.get("objective", "")
        self._quests: List[Dict[str, Any]] = []
        self._fail_conditions: List[List[Tuple[str, Tuple[str, ...]]]] = []
        for quest in self._spec.get("quests", []):
            win_events = quest.get("win_events", [])
            preconditions: List[Tuple[str, Tuple[str, ...]]] = []
            if win_events:
                cond = win_events[0].get("condition", {})
                for p in cond.get("preconditions", []):
                    args = tuple(a["name"] for a in p["arguments"])
                    preconditions.append((p["name"], args))
            if preconditions:  # skip empty quests
                self._quests.append({
                    "commands": quest.get("commands", []),
                    "reward": quest.get("reward", 1),
                    "preconditions": preconditions,
                })
            # Track fail events separately
            fail_events = quest.get("fail_events", [])
            for fe in fail_events:
                fc = fe.get("condition", {})
                fail_preconds: List[Tuple[str, Tuple[str, ...]]] = []
                for p in fc.get("preconditions", []):
                    args = tuple(a["name"] for a in p["arguments"])
                    fail_preconds.append((p["name"], args))
                if fail_preconds:
                    self._fail_conditions.append(fail_preconds)
        self._max_score = sum(q["reward"] for q in self._quests)
        if not self._objective and self._quests:
            self._objective = self._spec["quests"][0].get("desc", "")

        # Build initial state from the "world" facts
        self._initial_world = self._spec["world"]

        # Room connection maps: built once
        self._connections: Dict[str, Dict[str, str]] = {}  # room_id -> {direction -> room_id}
        self._door_links: Dict[Tuple[str, str], str] = {}  # (room_a, room_b) -> door_id
        self._key_matches: Dict[str, str] = {}  # key_id -> lockable_id (container or door)
        self._lockable_key: Dict[str, str] = {}  # lockable_id -> key_id

        self._build_static_maps()

        # Mutable state (set on reset)
        self._player_room: str = ""
        self._inventory: Set[str] = set()  # entity ids in inventory
        self._obj_locations: Dict[str, str] = {}  # obj_id -> room_id (for objects on floor)
        self._obj_in_container: Dict[str, str] = {}  # obj_id -> container_id
        self._obj_on_supporter: Dict[str, str] = {}  # obj_id -> supporter_id
        self._container_state: Dict[str, str] = {}  # container_id -> "open"/"closed"/"locked"
        self._door_state: Dict[str, str] = {}  # door_id -> "open"/"closed"/"locked"
        self._eaten: Set[str] = set()  # food items that have been eaten

        # Quest tracking
        self._score: int = 0
        self._won: bool = False
        self._done: bool = False
        self._moves: int = 0
        self._completed_quests: int = 0  # number of quests completed so far

    def _build_static_maps(self) -> None:
        """Build room connections, door links, and key matches from world facts."""
        direction_predicates = {
            "north_of": "north",
            "south_of": "south",
            "east_of": "east",
            "west_of": "west",
        }
        opposite = {"north": "south", "south": "north", "east": "west", "west": "east"}

        for fact in self._initial_world:
            name = fact["name"]
            args = [a["name"] for a in fact["arguments"]]

            if name in direction_predicates:
                # north_of(A, B) means A is north of B, so from B you go north to reach A
                direction = direction_predicates[name]
                room_from = args[1]
                room_to = args[0]
                self._connections.setdefault(room_from, {})[direction] = room_to

            elif name == "link":
                # link(room_a, door, room_b)
                room_a, door_id, room_b = args[0], args[1], args[2]
                self._door_links[(room_a, room_b)] = door_id

            elif name == "match":
                # match(key, lockable)
                key_id, lockable_id = args[0], args[1]
                self._key_matches[key_id] = lockable_id
                self._lockable_key[lockable_id] = key_id

    def reset(self) -> Tuple[str, Dict[str, Any]]:
        """Reset the game to initial state. Returns (observation, info)."""
        self._player_room = ""
        self._inventory.clear()
        self._obj_locations.clear()
        self._obj_in_container.clear()
        self._obj_on_supporter.clear()
        self._container_state.clear()
        self._door_state.clear()
        self._eaten.clear()
        self._score = 0
        self._won = False
        self._done = False
        self._moves = 1  # Inform7 counts reset/look as turn 1
        self._completed_quests = 0

        # Load state from world facts
        for fact in self._initial_world:
            name = fact["name"]
            args = [a["name"] for a in fact["arguments"]]
            arg_types = [a["type"] for a in fact["arguments"]]

            if name == "at":
                entity_id, room_id = args[0], args[1]
                etype = arg_types[0]
                if etype == "P":
                    self._player_room = room_id
                elif etype in ("o", "f", "k"):
                    self._obj_locations[entity_id] = room_id
                # containers (c) and supporters (s) are also "at" a room
                # but they are fixed in place -- we just note room for lookup
                elif etype in ("c", "s"):
                    self._obj_locations[entity_id] = room_id

            elif name == "in":
                entity_id, container_id = args[0], args[1]
                cont_type = arg_types[1]
                if cont_type == "I":
                    self._inventory.add(entity_id)
                elif cont_type == "c":
                    self._obj_in_container[entity_id] = container_id

            elif name == "on":
                entity_id, supporter_id = args[0], args[1]
                self._obj_on_supporter[entity_id] = supporter_id

            elif name == "open":
                entity_id = args[0]
                etype = arg_types[0]
                if etype == "c":
                    self._container_state[entity_id] = "open"
                elif etype == "d":
                    self._door_state[entity_id] = "open"

            elif name == "closed":
                entity_id = args[0]
                etype = arg_types[0]
                if etype == "c":
                    self._container_state[entity_id] = "closed"
                elif etype == "d":
                    self._door_state[entity_id] = "closed"

            elif name == "locked":
                entity_id = args[0]
                etype = arg_types[0]
                if etype == "c":
                    self._container_state[entity_id] = "locked"
                elif etype == "d":
                    self._door_state[entity_id] = "locked"

        # Generate initial observation: objective + look
        obs_parts = []
        obs_parts.append(self._objective)
        obs_parts.append("")
        obs_parts.append(self._generate_look())

        obs = "\n".join(obs_parts)
        info = self._make_info()
        return obs, info

    def step(self, command: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """
        Execute a command. Returns (observation, reward, done, info).

        The reward is 1.0 when all win conditions are satisfied (game won),
        0.0 otherwise (matching TextWorld's default behavior).
        """
        if self._done:
            return "The game has already ended.", 0.0, True, self._make_info()

        self._moves += 1
        command = command.strip()
        if not command:
            command = "look"

        obs = self._execute_command(command)

        # Check quest progress incrementally
        prev_score = self._score
        if not self._won and not self._done:
            self._check_quest_progress()
            # Compute score from actual quest reward values
            self._score = sum(
                self._quests[i]["reward"]
                for i in range(self._completed_quests)
            )
            # Inform7 prints "Your score has just gone up by N point(s)."
            if self._score > prev_score:
                delta = self._score - prev_score
                pts = "one point" if delta == 1 else f"{delta} points"
                obs += f"\n\n\nYour score has just gone up by {pts}."
            if self._completed_quests >= len(self._quests) and len(self._quests) > 0:
                self._won = True
                self._done = True
                obs += (
                    f"\n\n\n"
                    f"                                                       "
                    f"*** The End ***\n\n"
                    f"You scored {self._score} out of a possible {self._max_score}, "
                    f"in {self._moves} turns."
                )
            # Check fail conditions (e.g., eating the wrong item)
            elif self._check_fail_conditions():
                self._done = True
                obs += "\n\n*** You lost! ***"

        # TextWorld always returns reward = current score
        reward = float(self._score)

        info = self._make_info()
        return obs, reward, self._done, info

    def _make_info(self) -> Dict[str, Any]:
        """Build info dict matching TextWorld's game state interface."""
        return {
            "score": self._score,
            "max_score": self._max_score,
            "won": self._won,
            "moves": self._moves,
        }

    # ------------------------------------------------------------------
    # Win condition checking
    # ------------------------------------------------------------------

    def _check_quest_progress(self) -> int:
        """Check quests in order starting from last completed. Returns count of newly completed."""
        newly = 0
        while self._completed_quests < len(self._quests):
            quest = self._quests[self._completed_quests]
            if self._preconditions_met(quest["preconditions"]):
                self._completed_quests += 1
                newly += 1
            else:
                break
        return newly

    def _check_fail_conditions(self) -> bool:
        """Check if any fail condition is met. Returns True if game should end in failure."""
        for fail_preconds in self._fail_conditions:
            if self._preconditions_met(fail_preconds):
                return True
        return False

    def _preconditions_met(self, preconditions: List[Tuple[str, Tuple[str, ...]]]) -> bool:
        """Check if all preconditions of a quest are satisfied."""
        for pred_name, pred_args in preconditions:
            if pred_name == "event":
                continue
            if not self._fact_holds(pred_name, pred_args):
                return False
        return True

    def _fact_holds(self, pred_name: str, args: Tuple[str, ...]) -> bool:
        """Check if a single predicate holds in current state."""
        if pred_name == "at":
            entity = args[0]
            room = args[1]
            etype = self._get_type(entity)
            if etype == "P":
                return self._player_room == room
            else:
                return self._obj_locations.get(entity) == room

        elif pred_name == "in":
            entity = args[0]
            container = args[1]
            ctype = self._get_type(container)
            if ctype == "I" or container == "I":
                return entity in self._inventory
            else:
                return self._obj_in_container.get(entity) == container

        elif pred_name == "on":
            entity = args[0]
            supporter = args[1]
            return self._obj_on_supporter.get(entity) == supporter

        elif pred_name == "open":
            entity = args[0]
            etype = self._get_type(entity)
            if etype == "c":
                return self._container_state.get(entity) == "open"
            elif etype == "d":
                return self._door_state.get(entity) == "open"

        elif pred_name == "closed":
            entity = args[0]
            etype = self._get_type(entity)
            if etype == "c":
                return self._container_state.get(entity) == "closed"
            elif etype == "d":
                return self._door_state.get(entity) == "closed"

        elif pred_name == "locked":
            entity = args[0]
            etype = self._get_type(entity)
            if etype == "c":
                return self._container_state.get(entity) == "locked"
            elif etype == "d":
                return self._door_state.get(entity) == "locked"

        elif pred_name == "eaten":
            entity = args[0]
            return entity in self._eaten

        elif pred_name in ("north_of", "south_of", "east_of", "west_of", "free"):
            # Static structural facts — always true if set initially
            return True

        elif pred_name == "link":
            # link(room_a, door, room_b) — verify the specific link exists
            if len(args) == 3:
                room_a, door_id, room_b = args
                return self._door_links.get((room_a, room_b)) == door_id
            return True

        elif pred_name == "match":
            # match(key, lockable) — verify key matches
            if len(args) == 2:
                key_id, lockable_id = args
                return self._key_matches.get(key_id) == lockable_id
            return True

        return False

    def _get_type(self, entity_id: str) -> str:
        """Get the type code for an entity."""
        if entity_id == "P":
            return "P"
        if entity_id == "I":
            return "I"
        info = self._infos.get(entity_id)
        if info:
            return info.get("type", "")
        return ""

    # ------------------------------------------------------------------
    # Command parsing and execution
    # ------------------------------------------------------------------

    def _execute_command(self, command: str) -> str:
        """Parse and execute a single TextWorld command."""
        cmd_lower = command.lower().strip()

        # Navigation
        if cmd_lower in ("go north", "north", "n"):
            return self._do_go("north")
        elif cmd_lower in ("go south", "south", "s"):
            return self._do_go("south")
        elif cmd_lower in ("go east", "east", "e"):
            return self._do_go("east")
        elif cmd_lower in ("go west", "west", "w"):
            return self._do_go("west")

        # Look and inventory
        elif cmd_lower in ("look", "l"):
            return self._generate_look()
        elif cmd_lower in ("inventory", "i"):
            return self._do_inventory()

        # Examine
        elif cmd_lower.startswith("examine ") or cmd_lower.startswith("x "):
            target = command.split(None, 1)[1] if " " in command else ""
            return self._do_examine(target)

        # Take
        elif cmd_lower.startswith("take "):
            return self._parse_take(command[5:].strip())

        # Drop
        elif cmd_lower.startswith("drop "):
            target = command[5:].strip()
            return self._do_drop(target)

        # Open
        elif cmd_lower.startswith("open "):
            target = command[5:].strip()
            return self._do_open(target)

        # Close
        elif cmd_lower.startswith("close "):
            target = command[6:].strip()
            return self._do_close(target)

        # Unlock X with Y
        elif cmd_lower.startswith("unlock "):
            return self._parse_unlock(command[7:].strip())

        # Lock X with Y
        elif cmd_lower.startswith("lock "):
            return self._parse_lock(command[5:].strip())

        # Put X on Y
        elif cmd_lower.startswith("put "):
            return self._parse_put(command[4:].strip())

        # Insert X into Y
        elif cmd_lower.startswith("insert "):
            return self._parse_insert(command[7:].strip())

        # Eat
        elif cmd_lower.startswith("eat "):
            target = command[4:].strip()
            return self._do_eat(target)

        else:
            return "I don't understand that command."

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _do_go(self, direction: str) -> str:
        """Move player in a direction."""
        exits = self._connections.get(self._player_room, {})
        if direction not in exits:
            return "You can't go that way."

        target_room = exits[direction]

        # Check for door blocking the way
        door_id = (
            self._door_links.get((self._player_room, target_room))
            or self._door_links.get((target_room, self._player_room))
        )
        if door_id:
            state = self._door_state.get(door_id, "open")
            if state in ("locked", "closed"):
                return f"You have to open the {self._get_name(door_id)} first."

        self._player_room = target_room
        return self._generate_look()

    # ------------------------------------------------------------------
    # Look / room description
    # ------------------------------------------------------------------

    def _generate_look(self) -> str:
        """Generate the room description for current location."""
        room_info = self._infos.get(self._player_room, {})
        room_name = room_info.get("name", "unknown room")
        raw_desc = room_info.get("desc", "")

        # Process the Inform7 template in the description
        desc = self._process_template(raw_desc)

        lines = [f"-= {room_name.title()} =-", desc]

        # List objects on the floor in this room (portable objects only)
        floor_objects = self._get_floor_objects(self._player_room)
        if floor_objects:
            names = [self._get_name(oid) for oid in floor_objects]
            if len(names) == 1:
                a = self._article(names[0])
                lines.append(f"\nThere is {a} {names[0]} on the floor.")
            else:
                listing = self._english_list(names)
                lines.append(f"\nThere is {listing} on the floor.")

        return "\n".join(lines)

    def _get_floor_objects(self, room_id: str) -> List[str]:
        """Get portable objects on the floor of a room (not in containers/on supporters)."""
        result = []
        for obj_id, loc in self._obj_locations.items():
            if loc != room_id:
                continue
            etype = self._get_type(obj_id)
            if etype in ("o", "f", "k"):
                # Make sure it's not in a container or on a supporter
                if obj_id not in self._obj_in_container and obj_id not in self._obj_on_supporter:
                    result.append(obj_id)
        return result

    def _process_template(self, text: str, context_entity: Optional[str] = None) -> str:
        """Process Inform7-style conditional templates in room descriptions.

        context_entity: entity ID whose description is being processed.
            Used to resolve bare conditionals like [if open] in container/door descs.
        """
        if not text:
            return text

        # Process [if X is open/closed/locked] ... [else if ...] ... [otherwise] ... [end if]
        # and [a list of things in/on X] patterns
        result = self._resolve_inform7_conditionals(text, context_entity)
        return result

    def _resolve_inform7_conditionals(self, text: str, context_entity: Optional[str] = None) -> str:
        """Resolve Inform7 conditional blocks in text."""
        # We need to handle nested/sequential conditionals
        max_iters = 20
        for _ in range(max_iters):
            # Find the innermost [if ...] ... [end if] block
            match = re.search(
                r'\[if\s+(.+?)\](.*?)\[end if\]',
                text,
                re.DOTALL | re.IGNORECASE,
            )
            if not match:
                break

            full_match = match.group(0)
            condition_str = match.group(1)
            body = match.group(2)

            # Parse the body for [else if ...], [otherwise], [else] branches
            replacement = self._eval_conditional_block(condition_str, body, context_entity)
            text = text.replace(full_match, replacement, 1)

        # Now resolve [a list of things in X] / [a list of things on X]
        text = re.sub(
            r'\[a list of things in the (\w+)\]',
            lambda m: self._list_things_in(m.group(1)),
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r'\[a list of things in (\w+)\]',
            lambda m: self._list_things_in(m.group(1)),
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r'\[a list of things on the (\w+)\]',
            lambda m: self._list_things_on(m.group(1)),
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r'\[a list of things on (\w+)\]',
            lambda m: self._list_things_on(m.group(1)),
            text,
            flags=re.IGNORECASE,
        )

        # Inform7's Z-machine converts single-quoted words to double quotes
        # e.g., 'looking.' -> "looking."  (but not contractions like can't)
        text = re.sub(r"(?<=\s)'(\w+[.!?,;]?)'", r'"\1"', text)

        # Strip leading whitespace from each line (Inform7 does this)
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        return text.strip()

    def _eval_conditional_block(self, condition_str: str, body: str,
                               context_entity: Optional[str] = None) -> str:
        """Evaluate an if/else if/otherwise/else block."""
        # Split body by [else if ...], [otherwise], [else]
        branches: List[Tuple[str, str]] = []

        # The first branch is the if-branch
        parts = re.split(r'\[(?:else if|otherwise|else)\s*([^\]]*)\]', body, flags=re.IGNORECASE)

        if len(parts) == 1:
            # No else branches
            branches.append((condition_str, parts[0]))
        else:
            branches.append((condition_str, parts[0]))
            i = 1
            while i < len(parts):
                if i + 1 < len(parts):
                    cond = parts[i].strip()
                    text = parts[i + 1]
                    if not cond:
                        # [otherwise] or [else] -- always true fallback
                        cond = "__TRUE__"
                    branches.append((cond, text))
                    i += 2
                else:
                    branches.append(("__TRUE__", parts[i]))
                    i += 1

        for cond, text in branches:
            if cond == "__TRUE__" or self._eval_condition(cond, context_entity):
                return text

        return ""

    def _eval_condition(self, condition_str: str, context_entity: Optional[str] = None) -> bool:
        """Evaluate an Inform7 condition, including compound 'X and Y' conditions."""
        cond = condition_str.strip().lower()

        # Handle compound "A and B" conditions
        # Split on " and " but not inside "list of things in the X"
        and_parts = re.split(r'\s+and\s+', cond)
        if len(and_parts) > 1:
            return all(self._eval_single_condition(p.strip(), context_entity) for p in and_parts)

        return self._eval_single_condition(cond, context_entity)

    def _eval_single_condition(self, cond: str, context_entity: Optional[str] = None) -> bool:
        """Evaluate a single Inform7 condition."""

        # Bare "open"/"closed"/"locked" — refers to the context entity
        if cond in ("open", "closed", "locked") and context_entity:
            etype = self._get_type(context_entity)
            if etype == "c":
                actual = self._container_state.get(context_entity, "closed")
            elif etype == "d":
                actual = self._door_state.get(context_entity, "closed")
            else:
                return False
            # In Inform7, "locked" implies "closed" — a locked thing satisfies "is closed"
            if cond == "closed":
                return actual in ("closed", "locked")
            return actual == cond

        # "X is open/closed/locked"
        m = re.match(r'(\w+)\s+is\s+(open|closed|locked)', cond)
        if m:
            entity_ref = m.group(1)
            state = m.group(2)
            entity_id = self._resolve_entity_ref(entity_ref)
            if entity_id:
                etype = self._get_type(entity_id)
                if etype == "c":
                    actual = self._container_state.get(entity_id, "closed")
                elif etype == "d":
                    actual = self._door_state.get(entity_id, "closed")
                else:
                    return False
                # In Inform7, "locked" implies "closed"
                if state == "closed":
                    return actual in ("closed", "locked")
                return actual == state
            return False

        # "there is something in the X" / "there is something in X"
        m = re.match(r'there is something in\s+(?:the\s+)?(\w+)', cond)
        if m:
            entity_ref = m.group(1)
            entity_id = self._resolve_entity_ref(entity_ref)
            if entity_id:
                for obj, cont in self._obj_in_container.items():
                    if cont == entity_id:
                        return True
            return False

        # "there is something on the X" / "there is something on X"
        m = re.match(r'there is something on\s+(?:the\s+)?(\w+)', cond)
        if m:
            entity_ref = m.group(1)
            entity_id = self._resolve_entity_ref(entity_ref)
            if entity_id:
                for obj, sup in self._obj_on_supporter.items():
                    if sup == entity_id:
                        return True
            return False

        # "there is nothing on the X"
        m = re.match(r'there is nothing on\s+(?:the\s+)?(\w+)', cond)
        if m:
            entity_ref = m.group(1)
            entity_id = self._resolve_entity_ref(entity_ref)
            if entity_id:
                for obj, sup in self._obj_on_supporter.items():
                    if sup == entity_id:
                        return False
                return True
            return True

        # "the X contains nothing" / "X contains nothing"
        m = re.match(r'(?:the\s+)?(\w+)\s+contains\s+nothing', cond)
        if m:
            entity_ref = m.group(1)
            entity_id = self._resolve_entity_ref(entity_ref)
            if entity_id:
                for obj, cont in self._obj_in_container.items():
                    if cont == entity_id:
                        return False
                return True
            return True

        # Default: condition not recognized, return False
        return False

    def _resolve_entity_ref(self, ref: str) -> Optional[str]:
        """Resolve an entity reference (could be entity_id directly or a name)."""
        # Direct ID reference (e.g., c_0, d_1)
        if ref in self._infos:
            return ref
        # Name lookup
        ids = self._name_to_ids.get(ref.lower(), [])
        if ids:
            return ids[0]
        return None

    def _list_things_in(self, container_ref: str) -> str:
        """List objects inside a container."""
        container_id = self._resolve_entity_ref(container_ref)
        if not container_id:
            return "nothing"
        items = [oid for oid, cid in self._obj_in_container.items() if cid == container_id]
        if not items:
            return "nothing"
        names = [self._get_name(oid) for oid in items]
        return self._english_list(names)

    def _list_things_on(self, supporter_ref: str) -> str:
        """List objects on a supporter."""
        supporter_id = self._resolve_entity_ref(supporter_ref)
        if not supporter_id:
            return "nothing"
        items = [oid for oid, sid in self._obj_on_supporter.items() if sid == supporter_id]
        if not items:
            return "nothing"
        names = [self._get_name(oid) for oid in items]
        return self._english_list(names)

    @staticmethod
    def _article(name: str) -> str:
        """Return 'a' or 'an' depending on the first letter."""
        return "an" if name and name[0].lower() in "aeiou" else "a"

    @classmethod
    def _english_list(cls, names: List[str]) -> str:
        """Format a list of names as English: 'a X, an Y and a Z'."""
        if not names:
            return "nothing"
        articles = [f"{cls._article(n)} {n}" for n in names]
        if len(articles) == 1:
            return articles[0]
        return ", ".join(articles[:-1]) + " and " + articles[-1]

    # ------------------------------------------------------------------
    # Inventory
    # ------------------------------------------------------------------

    def _do_inventory(self) -> str:
        if not self._inventory:
            return "You are carrying nothing."
        names = [self._get_name(oid) for oid in sorted(self._inventory)]
        listing = self._english_list(names)
        return f"You are carrying: {listing}."

    # ------------------------------------------------------------------
    # Examine
    # ------------------------------------------------------------------

    def _do_examine(self, target: str) -> str:
        entity_id = self._resolve_target(target)
        if not entity_id:
            return "You can't see any such thing."

        info = self._infos.get(entity_id, {})
        desc = info.get("desc", "")
        etype = self._get_type(entity_id)

        if desc:
            desc = self._process_template(desc, context_entity=entity_id)

        if etype == "c":
            state = self._container_state.get(entity_id, "closed")
            if not desc:
                name = self._get_name(entity_id)
                desc = f"The {name} is {state}."
            if state == "open":
                items_in = [oid for oid, cid in self._obj_in_container.items() if cid == entity_id]
                if items_in:
                    names = [self._get_name(oid) for oid in items_in]
                    desc += f"\nThe {self._get_name(entity_id)} contains {self._english_list(names)}."
                else:
                    desc += f"\nThe {self._get_name(entity_id)} is empty."
        elif etype == "s":
            if not desc:
                name = self._get_name(entity_id)
                desc = f"The {name} is here."
            items_on = [oid for oid, sid in self._obj_on_supporter.items() if sid == entity_id]
            if items_on:
                names = [self._get_name(oid) for oid in items_on]
                desc += f"\nOn the {self._get_name(entity_id)} you can see {self._english_list(names)}."
            else:
                desc += f"\nThe {self._get_name(entity_id)} is empty."
        elif etype == "d":
            state = self._door_state.get(entity_id, "closed")
            if not desc:
                name = self._get_name(entity_id)
                desc = f"The {name} is {state}."
        else:
            if not desc:
                desc = f"You see nothing special about the {self._get_name(entity_id)}."

        return desc

    # ------------------------------------------------------------------
    # Take (with variants: take X, take X from Y)
    # ------------------------------------------------------------------

    def _parse_take(self, args_str: str) -> str:
        """Parse 'take X' or 'take X from Y'."""
        # Check for "take X from Y"
        m = re.match(r'(.+?)\s+from\s+(.+)', args_str, re.IGNORECASE)
        if m:
            obj_name = m.group(1).strip()
            source_name = m.group(2).strip()
            return self._do_take_from(obj_name, source_name)
        else:
            return self._do_take(args_str)

    def _do_take(self, target: str) -> str:
        """Take an object from the floor or from an open container/supporter in the room."""
        entity_id = self._resolve_target_in_room(target)
        if not entity_id:
            return "You can't see any such thing."

        etype = self._get_type(entity_id)
        if etype not in ("o", "f", "k"):
            return f"You can't take the {self._get_name(entity_id)}."

        # Object on floor?
        if self._obj_locations.get(entity_id) == self._player_room and \
                entity_id not in self._obj_in_container and \
                entity_id not in self._obj_on_supporter:
            del self._obj_locations[entity_id]
            self._inventory.add(entity_id)
            return f"You pick up the {self._get_name(entity_id)} from the ground."

        # Object on a supporter in this room?
        if entity_id in self._obj_on_supporter:
            sup_id = self._obj_on_supporter[entity_id]
            if self._obj_locations.get(sup_id) == self._player_room:
                del self._obj_on_supporter[entity_id]
                self._inventory.add(entity_id)
                return f"You take the {self._get_name(entity_id)} from the {self._get_name(sup_id)}."

        # Object in an open container in this room?
        if entity_id in self._obj_in_container:
            cont_id = self._obj_in_container[entity_id]
            if self._obj_locations.get(cont_id) == self._player_room:
                state = self._container_state.get(cont_id, "closed")
                if state == "open":
                    del self._obj_in_container[entity_id]
                    self._inventory.add(entity_id)
                    return f"You take the {self._get_name(entity_id)} from the {self._get_name(cont_id)}."
                else:
                    return f"The {self._get_name(cont_id)} is {state}."

        return f"You can't take the {self._get_name(entity_id)}."

    def _do_take_from(self, obj_name: str, source_name: str) -> str:
        """Take an object from a specific container or supporter."""
        source_id = self._resolve_target_in_room(source_name)
        if not source_id:
            return "You can't see any such thing."

        source_type = self._get_type(source_id)

        if source_type == "s":
            # Take from supporter
            obj_id = self._find_obj_on_supporter(obj_name, source_id)
            if not obj_id:
                return "You can't see any such thing."
            del self._obj_on_supporter[obj_id]
            self._inventory.add(obj_id)
            return f"You take the {self._get_name(obj_id)} from the {self._get_name(source_id)}."

        elif source_type == "c":
            state = self._container_state.get(source_id, "closed")
            if state != "open":
                return f"The {self._get_name(source_id)} is {state}."
            obj_id = self._find_obj_in_container(obj_name, source_id)
            if not obj_id:
                return "You can't see any such thing."
            del self._obj_in_container[obj_id]
            self._inventory.add(obj_id)
            return f"You take the {self._get_name(obj_id)} from the {self._get_name(source_id)}."

        return f"You can't take things from the {self._get_name(source_id)}."

    # ------------------------------------------------------------------
    # Drop
    # ------------------------------------------------------------------

    def _do_drop(self, target: str) -> str:
        entity_id = self._resolve_target_in_inventory(target)
        if not entity_id:
            return "You can't see any such thing."
        self._inventory.discard(entity_id)
        self._obj_locations[entity_id] = self._player_room
        return f"You drop the {self._get_name(entity_id)} on the ground."

    # ------------------------------------------------------------------
    # Open / Close
    # ------------------------------------------------------------------

    def _do_open(self, target: str) -> str:
        entity_id = self._resolve_target_in_room_or_adjacent(target)
        if not entity_id:
            return "You can't see any such thing."

        etype = self._get_type(entity_id)

        if etype == "c":
            if self._obj_locations.get(entity_id) != self._player_room:
                return "You can't see any such thing."
            state = self._container_state.get(entity_id, "closed")
            if state == "open":
                return f"The {self._get_name(entity_id)} is already open."
            elif state == "locked":
                return f"The {self._get_name(entity_id)} is locked."
            self._container_state[entity_id] = "open"
            # Show what's inside — Inform7 says "revealing" not "contains"
            items_in = [oid for oid, cid in self._obj_in_container.items() if cid == entity_id]
            name = self._get_name(entity_id)
            if items_in:
                names = [self._get_name(oid) for oid in items_in]
                return f"You open the {name}, revealing {self._english_list(names)}."
            else:
                return f"You open the {name}."

        elif etype == "d":
            state = self._door_state.get(entity_id, "closed")
            dn = self._the_name(entity_id)
            if state == "open":
                return f"That's already open."
            elif state == "locked":
                return f"It is locked."
            self._door_state[entity_id] = "open"
            return f"You open {dn}."

        return f"You can't open {self._the_name(entity_id)}."

    def _do_close(self, target: str) -> str:
        entity_id = self._resolve_target_in_room_or_adjacent(target)
        if not entity_id:
            return "You can't see any such thing."

        etype = self._get_type(entity_id)

        if etype == "c":
            if self._obj_locations.get(entity_id) != self._player_room:
                return "You can't see any such thing."
            state = self._container_state.get(entity_id, "closed")
            if state == "closed":
                return f"The {self._get_name(entity_id)} is already closed."
            elif state == "locked":
                return f"The {self._get_name(entity_id)} is locked."
            self._container_state[entity_id] = "closed"
            return f"You close the {self._get_name(entity_id)}."

        elif etype == "d":
            state = self._door_state.get(entity_id, "closed")
            dn = self._the_name(entity_id)
            if state == "closed":
                return f"That's already closed."
            elif state == "locked":
                return f"It is locked."
            self._door_state[entity_id] = "closed"
            return f"You close {dn}."

        return f"You can't close {self._the_name(entity_id)}."

    # ------------------------------------------------------------------
    # Unlock / Lock
    # ------------------------------------------------------------------

    def _parse_unlock(self, args_str: str) -> str:
        m = re.match(r'(.+?)\s+with\s+(.+)', args_str, re.IGNORECASE)
        if not m:
            return "You need to specify what to unlock it with. Try: unlock <thing> with <key>"
        target_name = m.group(1).strip()
        key_name = m.group(2).strip()
        return self._do_unlock(target_name, key_name)

    def _do_unlock(self, target_name: str, key_name: str) -> str:
        target_id = self._resolve_target_in_room_or_adjacent(target_name)
        if not target_id:
            return "You can't see any such thing."

        key_id = self._resolve_target_in_inventory(key_name)
        if not key_id:
            return "You can't see any such thing."

        etype = self._get_type(target_id)

        if etype in ("c", "d"):
            state = self._container_state.get(target_id) if etype == "c" else self._door_state.get(target_id)
            if state != "locked":
                return f"The {self._get_name(target_id)} is not locked."

            # Check key matches
            expected_key = self._lockable_key.get(target_id)
            if expected_key != key_id:
                return f"The {self._get_name(key_id)} doesn't fit the {self._get_name(target_id)}."

            if etype == "c":
                self._container_state[target_id] = "closed"
            else:
                self._door_state[target_id] = "closed"
            return f"You unlock {self._the_name(target_id)}."

        return f"You can't unlock {self._the_name(target_id)}."

    def _parse_lock(self, args_str: str) -> str:
        m = re.match(r'(.+?)\s+with\s+(.+)', args_str, re.IGNORECASE)
        if not m:
            return "You need to specify what to lock it with. Try: lock <thing> with <key>"
        target_name = m.group(1).strip()
        key_name = m.group(2).strip()
        return self._do_lock(target_name, key_name)

    def _do_lock(self, target_name: str, key_name: str) -> str:
        target_id = self._resolve_target_in_room_or_adjacent(target_name)
        if not target_id:
            return "You can't see any such thing."

        key_id = self._resolve_target_in_inventory(key_name)
        if not key_id:
            return "You can't see any such thing."

        etype = self._get_type(target_id)

        if etype in ("c", "d"):
            state = self._container_state.get(target_id) if etype == "c" else self._door_state.get(target_id)
            if state == "locked":
                return f"The {self._get_name(target_id)} is already locked."
            if state == "open":
                return f"First you would have to close {self._the_name(target_id)}."
            if state != "closed":
                return f"The {self._get_name(target_id)} is not closed."

            expected_key = self._lockable_key.get(target_id)
            if expected_key != key_id:
                return f"The {self._get_name(key_id)} doesn't fit the {self._get_name(target_id)}."

            if etype == "c":
                self._container_state[target_id] = "locked"
            else:
                self._door_state[target_id] = "locked"
            return f"You lock {self._the_name(target_id)}."

        return f"You can't lock {self._the_name(target_id)}."

    # ------------------------------------------------------------------
    # Put / Insert
    # ------------------------------------------------------------------

    def _parse_put(self, args_str: str) -> str:
        m = re.match(r'(.+?)\s+on\s+(.+)', args_str, re.IGNORECASE)
        if not m:
            return "You need to specify where to put it. Try: put <item> on <supporter>"
        obj_name = m.group(1).strip()
        sup_name = m.group(2).strip()
        return self._do_put(obj_name, sup_name)

    def _do_put(self, obj_name: str, sup_name: str) -> str:
        obj_id = self._resolve_target_in_inventory(obj_name)
        if not obj_id:
            return "You can't see any such thing."

        sup_id = self._resolve_target_in_room(sup_name)
        if not sup_id:
            return "You can't see any such thing."

        if self._get_type(sup_id) != "s":
            return f"You can't put things on the {self._get_name(sup_id)}."

        self._inventory.discard(obj_id)
        self._obj_on_supporter[obj_id] = sup_id
        return f"You put the {self._get_name(obj_id)} on the {self._get_name(sup_id)}."

    def _parse_insert(self, args_str: str) -> str:
        m = re.match(r'(.+?)\s+into\s+(.+)', args_str, re.IGNORECASE)
        if not m:
            return "You need to specify where to insert it. Try: insert <item> into <container>"
        obj_name = m.group(1).strip()
        cont_name = m.group(2).strip()
        return self._do_insert(obj_name, cont_name)

    def _do_insert(self, obj_name: str, cont_name: str) -> str:
        obj_id = self._resolve_target_in_inventory(obj_name)
        if not obj_id:
            return "You can't see any such thing."

        cont_id = self._resolve_target_in_room(cont_name)
        if not cont_id:
            return "You can't see any such thing."

        if self._get_type(cont_id) != "c":
            return f"You can't insert things into the {self._get_name(cont_id)}."

        state = self._container_state.get(cont_id, "closed")
        if state != "open":
            return f"The {self._get_name(cont_id)} is {state}."

        self._inventory.discard(obj_id)
        self._obj_in_container[obj_id] = cont_id
        # Inform7 says "put into" not "insert into"
        return f"You put the {self._get_name(obj_id)} into the {self._get_name(cont_id)}."

    # ------------------------------------------------------------------
    # Eat
    # ------------------------------------------------------------------

    def _do_eat(self, target: str) -> str:
        """Eat a food item from inventory."""
        entity_id = self._resolve_target_in_inventory(target)
        if not entity_id:
            return "You can't see any such thing."

        etype = self._get_type(entity_id)
        if etype != "f":
            return f"You can't eat the {self._get_name(entity_id)}."

        self._inventory.discard(entity_id)
        self._eaten.add(entity_id)
        return f"You eat the {self._get_name(entity_id)}. Not bad."

    # ------------------------------------------------------------------
    # Entity resolution helpers
    # ------------------------------------------------------------------

    def _get_name(self, entity_id: str) -> str:
        """Get the display name for an entity."""
        info = self._infos.get(entity_id, {})
        return info.get("name", entity_id) or entity_id

    def _the_name(self, entity_id: str) -> str:
        """Get 'the X' or bare name for doors (matching Inform7 behavior)."""
        name = self._get_name(entity_id)
        if self._get_type(entity_id) == "d":
            return name  # Inform7 uses bare name for doors
        return f"the {name}"

    def _resolve_target(self, name: str) -> Optional[str]:
        """Resolve a name to entity ID. Checks room, inventory, adjacent doors."""
        result = self._resolve_target_in_room(name)
        if result:
            return result
        result = self._resolve_target_in_inventory(name)
        if result:
            return result
        result = self._resolve_adjacent_door(name)
        if result:
            return result
        return None

    def _resolve_target_in_room(self, name: str) -> Optional[str]:
        """Resolve a name to entity ID, considering only things in the current room."""
        name_lower = name.lower().strip()
        if not name_lower:
            return None

        # Collect all entities in the current room
        candidates: List[str] = []
        for eid, loc in self._obj_locations.items():
            if loc == self._player_room:
                candidates.append(eid)
        # Also objects in containers/on supporters that are in this room
        for eid, cid in self._obj_in_container.items():
            if self._obj_locations.get(cid) == self._player_room:
                candidates.append(eid)
        for eid, sid in self._obj_on_supporter.items():
            if self._obj_locations.get(sid) == self._player_room:
                candidates.append(eid)

        return self._match_entity(name_lower, candidates)

    def _resolve_target_in_room_or_adjacent(self, name: str) -> Optional[str]:
        """Resolve in room first, then try adjacent doors."""
        result = self._resolve_target_in_room(name)
        if result:
            return result
        return self._resolve_adjacent_door(name)

    def _resolve_adjacent_door(self, name: str) -> Optional[str]:
        """Resolve a name to a door adjacent to the current room."""
        name_lower = name.lower().strip()
        if not name_lower:
            return None

        # Find all doors connected to current room
        door_candidates: List[str] = []
        for (ra, rb), did in self._door_links.items():
            if ra == self._player_room or rb == self._player_room:
                if did not in door_candidates:
                    door_candidates.append(did)

        return self._match_entity(name_lower, door_candidates)

    def _resolve_target_in_inventory(self, name: str) -> Optional[str]:
        """Resolve a name to entity ID, considering only inventory."""
        name_lower = name.lower().strip()
        if not name_lower:
            return None
        return self._match_entity(name_lower, list(self._inventory))

    def _match_entity(self, name_lower: str, candidates: List[str]) -> Optional[str]:
        """
        Match a name string against a list of candidate entity IDs.

        Uses exact name match first, then noun match, then partial/substring match.
        Case-insensitive throughout.
        """
        if not name_lower or not candidates:
            return None

        # Strip articles
        for article in ("the ", "a ", "an "):
            if name_lower.startswith(article):
                name_lower = name_lower[len(article):]
                break

        # Direct entity ID match
        if name_lower in self._infos and name_lower in candidates:
            return name_lower

        # Score candidates
        best_id: Optional[str] = None
        best_score: int = 0

        for eid in candidates:
            info = self._infos.get(eid, {})
            ename = (info.get("name") or "").lower()
            enoun = (info.get("noun") or "").lower()
            eadj = (info.get("adj") or "").lower()
            full_name = f"{eadj} {enoun}".strip() if eadj else enoun

            score = 0

            # Exact full name match (highest priority)
            if ename == name_lower or full_name == name_lower:
                score = 100
            # Exact noun match
            elif enoun == name_lower:
                score = 80
            # Query is a prefix of name (e.g. "chest" matches "chest drawer")
            elif ename.startswith(name_lower):
                score = 60
            # Noun prefix match
            elif enoun and enoun.startswith(name_lower):
                score = 50
            # Query contains exact name as a word (e.g. "old key" matches "key")
            # But NOT "cuboid box" matching "box" — require word boundary
            elif name_lower in ename:
                score = 40

            if score > best_score:
                best_score = score
                best_id = eid

        return best_id

    def _find_obj_in_container(self, obj_name: str, container_id: str) -> Optional[str]:
        """Find an object inside a specific container by name."""
        name_lower = obj_name.lower().strip()
        for article in ("the ", "a ", "an "):
            if name_lower.startswith(article):
                name_lower = name_lower[len(article):]
                break
        candidates = [oid for oid, cid in self._obj_in_container.items() if cid == container_id]
        return self._match_entity(name_lower, candidates)

    def _find_obj_on_supporter(self, obj_name: str, supporter_id: str) -> Optional[str]:
        """Find an object on a specific supporter by name."""
        name_lower = obj_name.lower().strip()
        for article in ("the ", "a ", "an "):
            if name_lower.startswith(article):
                name_lower = name_lower[len(article):]
                break
        candidates = [oid for oid, sid in self._obj_on_supporter.items() if sid == supporter_id]
        return self._match_entity(name_lower, candidates)

    # ------------------------------------------------------------------
    # Properties for external access
    # ------------------------------------------------------------------

    @property
    def score(self) -> int:
        return self._score

    @property
    def won(self) -> bool:
        return self._won

    @property
    def done(self) -> bool:
        return self._done

    @property
    def feedback(self) -> str:
        """Alias for compatibility with TextWorld game state interface."""
        return self._last_obs if hasattr(self, "_last_obs") else ""

    @property
    def moves(self) -> int:
        return self._moves
