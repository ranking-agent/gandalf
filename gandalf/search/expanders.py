"""Predicate and qualifier expansion using BMT hierarchy."""

from bmt.toolkit import Toolkit


class QualifierExpander:
    """Handles qualifier value expansion using BMT hierarchy at query time.

    This class caches BMT lookups for performance and provides methods to expand
    qualifier values to include their descendants, following the reasoner-transpiler
    approach.

    For example, if a query specifies qualifier_value "activity_or_abundance",
    this will expand to also match "activity" and "abundance" (the child values).
    """

    def __init__(self, bmt: Toolkit):
        self.bmt = bmt
        self._descendants_cache: dict[tuple[str, str], list[str]] = {}
        self._enum_names: list[str] | None = None

    def _get_enum_names(self) -> list[str]:
        """Get all enum names from the Biolink Model (cached)."""
        if self._enum_names is None:
            try:
                self._enum_names = list(self.bmt.view.all_enums().keys())
            except Exception:
                self._enum_names = []
        return self._enum_names

    def get_value_descendants(self, qualifier_value: str) -> list[str]:
        """Get all descendant values for a qualifier value across all Biolink enums.

        Args:
            qualifier_value: The qualifier value to expand

        Returns:
            List of descendant values (including the original value)
        """
        # Check cache with just the value as key (we search all enums)
        cache_key = ("_all_", qualifier_value)
        if cache_key in self._descendants_cache:
            return self._descendants_cache[cache_key]

        descendants = set()
        descendants.add(qualifier_value)  # Always include the original value

        # Search all enums for this value and get descendants
        for enum_name in self._get_enum_names():
            try:
                if self.bmt.is_permissible_value_of_enum(
                    enum_name=enum_name, value=qualifier_value
                ):
                    enum_descendants = self.bmt.get_permissible_value_descendants(
                        permissible_value=qualifier_value, enum_name=enum_name
                    )
                    if enum_descendants:
                        descendants.update(enum_descendants)
            except Exception:
                # If BMT methods fail, continue with other enums
                continue

        result = list(descendants)
        self._descendants_cache[cache_key] = result
        return result

    def expand_qualifier_constraints(
        self, qualifier_constraints: list[dict]
    ) -> list[dict]:
        """Expand qualifier constraints to include descendant values.

        This transforms each qualifier in a qualifier_set by expanding its value
        to include descendant values. The result uses a special format where each
        qualifier has "qualifier_values" (plural) containing all acceptable values.

        The matching semantics remain:
        - OR between qualifier_sets (edge matches if ANY set matches)
        - AND within each qualifier_set (edge must match ALL qualifiers in a set)
        - OR between expanded values (edge matches if it has ANY of the descendant values)

        Args:
            qualifier_constraints: List of qualifier constraint dicts, each with
                                   a 'qualifier_set' containing qualifiers to match

        Returns:
            Expanded qualifier constraints with "qualifier_values" lists
        """
        if not qualifier_constraints:
            return qualifier_constraints

        expanded_constraints = []
        for constraint in qualifier_constraints:
            qualifier_set = constraint.get("qualifier_set", [])
            if not qualifier_set:
                # Empty qualifier_set matches any edge, keep as-is
                expanded_constraints.append(constraint)
                continue

            # Expand each qualifier in the set
            expanded_qualifiers = []
            for qualifier in qualifier_set:
                type_id = qualifier.get("qualifier_type_id")
                value = qualifier.get("qualifier_value")

                if not type_id or not value:
                    # Keep original if missing fields
                    expanded_qualifiers.append(qualifier)
                    continue

                # Get descendant values (includes original)
                descendant_values = self.get_value_descendants(value)

                # Create expanded qualifier with list of acceptable values
                expanded_qualifiers.append(
                    {
                        "qualifier_type_id": type_id,
                        "qualifier_values": descendant_values,  # plural - list of values
                    }
                )

            expanded_constraints.append({"qualifier_set": expanded_qualifiers})

        return expanded_constraints


class PredicateExpander:
    """Handles predicate expansion for symmetric and inverse predicates at query time.

    This class caches BMT lookups for performance and provides methods to determine
    what predicates should match when traversing edges in different directions.

    Predicate handling follows the reasoner-transpiler rules:
    1. If 'biolink:related_to' is queried, treat as "any predicate"
    2. For each predicate P:
       - Get inverse Q (if exists) -> add to inverse predicates
       - If P is symmetric -> add P to inverse predicates
    3. Expand both predicates and inverse predicates to descendants
    4. Filter descendants to only those that are canonical OR symmetric

    For a query predicate P:
    - Symmetric predicates: If P is symmetric, an edge A--P-->B also represents B--P-->A
    - Inverse predicates: If P has inverse Q, an edge A--P-->B is equivalent to B--Q-->A

    When traversing:
    - Forward (outgoing edges): Match predicate P directly
    - Backward (incoming edges): Match P if symmetric, or match inverse(P) if it exists
    """

    def __init__(self, bmt: Toolkit):
        self.bmt = bmt
        self._inverse_cache: dict[str, str | None] = {}
        self._symmetric_cache: dict[str, bool] = {}
        self._canonical_cache: dict[str, bool] = {}
        self._descendants_cache: dict[str, list[str]] = {}

    def is_symmetric(self, predicate: str) -> bool:
        """Check if a predicate is symmetric (cached)."""
        if predicate not in self._symmetric_cache:
            try:
                self._symmetric_cache[predicate] = self.bmt.is_symmetric(predicate)
            except Exception:
                self._symmetric_cache[predicate] = False
        return self._symmetric_cache[predicate]

    def is_canonical(self, predicate: str) -> bool:
        """Check if a predicate is canonical (cached).

        A predicate is canonical if it has the 'canonical_predicate' annotation
        set to True in the Biolink Model.
        """
        if predicate not in self._canonical_cache:
            try:
                element = self.bmt.get_element(predicate)
                if element is None:
                    self._canonical_cache[predicate] = False
                else:
                    # Check for canonical_predicate annotation
                    annotations = getattr(element, "annotations", {}) or {}
                    self._canonical_cache[predicate] = bool(
                        annotations.get("canonical_predicate", False)
                    )
            except Exception:
                self._canonical_cache[predicate] = False
        return self._canonical_cache[predicate]

    def is_canonical_or_symmetric(self, predicate: str) -> bool:
        """Check if a predicate is either canonical or symmetric."""
        return self.is_canonical(predicate) or self.is_symmetric(predicate)

    def get_inverse(self, predicate: str) -> str | None:
        """Get the inverse of a predicate if one exists (cached)."""
        if predicate not in self._inverse_cache:
            try:
                if self.bmt.has_inverse(predicate):
                    inverse = self.bmt.get_inverse_predicate(predicate, formatted=True)
                    self._inverse_cache[predicate] = inverse
                else:
                    self._inverse_cache[predicate] = None
            except Exception:
                self._inverse_cache[predicate] = None
        return self._inverse_cache[predicate]

    def get_descendants(self, predicate: str) -> list[str]:
        """Get all descendants of a predicate (cached)."""
        if predicate not in self._descendants_cache:
            try:
                element = self.bmt.get_element(predicate)
                if element is None:
                    self._descendants_cache[predicate] = []
                else:
                    self._descendants_cache[predicate] = self.bmt.get_descendants(
                        predicate, formatted=True
                    )
            except Exception:
                self._descendants_cache[predicate] = []
        return self._descendants_cache[predicate]

    def get_filtered_descendants(self, predicate: str) -> list[str]:
        """Get descendants of a predicate, filtered to only canonical OR symmetric.

        This follows the reasoner-transpiler behavior where only predicates that
        are either marked as canonical_predicate or are symmetric are included
        in query expansion.
        """
        descendants = self.get_descendants(predicate)
        return [d for d in descendants if self.is_canonical_or_symmetric(d)]

    def expand_predicates(
        self, predicates: list[str]
    ) -> tuple[list[str], list[str] | None]:
        """Expand predicates following reasoner-transpiler rules.

        This method:
        1. Handles 'biolink:related_to' as "any predicate" (returns empty lists)
        2. For each predicate, gets its inverse and adds to inverse list
        3. For symmetric predicates, adds them to the inverse list too
        4. Expands both lists to descendants
        5. Filters to only canonical OR symmetric predicates

        Args:
            predicates: List of predicate CURIEs from the query

        Returns:
            Tuple of (forward_predicates, inverse_predicates) where:
            - forward_predicates: Predicates to match in the forward direction
              (empty list means match all)
            - inverse_predicates: Predicates to match in the reverse direction
              (empty list means match all, None means don't check inverse)
        """
        # Handle 'related_to' or no predicates as "any predicate" in both directions
        if not predicates or "biolink:related_to" in predicates:
            return [], []

        # Collect inverse predicates
        inverse_preds = []
        for pred in predicates:
            # Get explicit inverse
            inverse = self.get_inverse(pred)
            if inverse:
                inverse_preds.append(inverse)
            # Symmetric predicates are their own inverse for bidirectional matching
            if self.is_symmetric(pred):
                inverse_preds.append(pred)

        # Expand to descendants and filter to canonical/symmetric
        # Always include the original query predicates (they should always match)
        # Only filter descendants to canonical/symmetric
        forward_expanded = list(predicates)
        for pred in predicates:
            forward_expanded.extend(self.get_filtered_descendants(pred))

        # Always include the original inverse predicates
        inverse_expanded = list(inverse_preds)
        for pred in inverse_preds:
            inverse_expanded.extend(self.get_filtered_descendants(pred))

        # Deduplicate while preserving order
        forward_unique = list(dict.fromkeys(forward_expanded))
        inverse_unique = list(dict.fromkeys(inverse_expanded))

        # Return None for inverse when there are no inverse predicates to check,
        # to distinguish from the empty-list wildcard used by related_to
        return forward_unique, inverse_unique if inverse_unique else None

    def get_predicates_for_incoming_edges(self, predicates: list[str]) -> set[str]:
        """Get predicates that should match on incoming edges.

        When we're looking for edges with predicate P pointing TO a node,
        we should also consider:
        - P itself if it's stored in the incoming direction
        - The inverse of P, since an incoming edge with inverse(P) represents
          the same relationship as an outgoing edge with P

        Args:
            predicates: List of predicates we're searching for

        Returns:
            Set of predicates to match on incoming edges
        """
        result = set()
        for pred in predicates:
            # Always include the original predicate for direct matches
            result.add(pred)
            # If P has inverse Q, then an incoming edge with Q is equivalent
            # to an outgoing edge with P from the perspective of the target node
            inverse = self.get_inverse(pred)
            if inverse:
                result.add(inverse)
        return result

    def get_predicates_for_outgoing_edges(self, predicates: list[str]) -> set[str]:
        """Get predicates that should match on outgoing edges.

        When we're looking for edges with predicate P pointing FROM a node,
        we should also consider:
        - P itself for direct matches
        - The inverse of P when checking from the object's perspective

        Args:
            predicates: List of predicates we're searching for

        Returns:
            Set of predicates to match on outgoing edges
        """
        result = set()
        for pred in predicates:
            result.add(pred)
            # Include inverse for bidirectional matching
            inverse = self.get_inverse(pred)
            if inverse:
                result.add(inverse)
        return result

    def should_check_reverse_direction(self, predicate: str) -> bool:
        """Determine if we should also check the reverse direction for this predicate.

        Returns True if the predicate is symmetric or has an inverse defined.
        """
        return self.is_symmetric(predicate) or self.get_inverse(predicate) is not None

    def get_reverse_predicate(self, predicate: str) -> str | None:
        """Get the predicate to use when checking the reverse direction.

        For symmetric predicates, returns the same predicate.
        For predicates with inverses, returns the inverse.
        For other predicates, returns None.
        """
        if self.is_symmetric(predicate):
            return predicate
        return self.get_inverse(predicate)
