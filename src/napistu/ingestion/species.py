from typing import Dict, Union, Set, List, Any

from napistu.ingestion.constants import LATIN_TO_COMMON_SPECIES_NAMES


class SpeciesValidator:
    """
    A class for validating and converting between common and Latin species names.

    Accepts either common names (e.g., 'human') or Latin names (e.g., 'Homo sapiens')
    and provides access to both forms through attributes.

    Parameters
    ----------
    species_input : str
        Either a common species name (e.g., 'human') or Latin species name
        (e.g., 'Homo sapiens'). Case-insensitive.

    Attributes
    ----------
    latin_name : str
        The Latin species name (e.g., 'Homo sapiens').
    common_name : str
        The common species name (e.g., 'human').

    Raises
    ------
    ValueError
        If the provided species_input is not recognized or is not a string.

    Examples
    --------
    >>> species = SpeciesValidator("Homo sapiens")
    >>> species.common_name
    'human'
    >>> species.latin_name
    'Homo sapiens'

    >>> species = SpeciesValidator("mouse")
    >>> species.latin_name
    'Mus musculus'

    >>> species = SpeciesValidator("HUMAN")  # case-insensitive
    >>> species.latin_name
    'Homo sapiens'
    """

    def __init__(self, species_input: str) -> None:
        """
        Initialize with either common or Latin species name.

        Parameters
        ----------
        species_input : str
            Either common name (e.g., 'human') or Latin species name
            (e.g., 'Homo sapiens'). Case-insensitive.

        Raises
        ------
        ValueError
            If species_input is not recognized or is not a string.
        """
        self._latin_name = None
        self._common_name = None
        self._validate_and_set_species(species_input)

    def _validate_and_set_species(self, species_input: str) -> None:
        """
        Validate input and set both Latin and common names.

        Parameters
        ----------
        species_input : str
            The species name to validate and normalize.

        Raises
        ------
        ValueError
            If species_input is not a string or not found in known species.
        """
        if not isinstance(species_input, str):
            raise ValueError("Species input must be a string")

        # Normalize input (strip whitespace, handle case variations)
        normalized_input = species_input.strip()

        # Create reverse lookup (common name -> Latin name)
        common_to_latin = {
            common: latin for latin, common in LATIN_TO_COMMON_SPECIES_NAMES.items()
        }

        # Check if input is a Latin name (case-insensitive)
        latin_match = None
        for latin_name in LATIN_TO_COMMON_SPECIES_NAMES.keys():
            if normalized_input.lower() == latin_name.lower():
                latin_match = latin_name
                break

        if latin_match:
            self._latin_name = latin_match
            self._common_name = LATIN_TO_COMMON_SPECIES_NAMES[latin_match]
            return

        # Check if input is a common name (case-insensitive)
        common_match = None
        for common_name in common_to_latin.keys():
            if normalized_input.lower() == common_name.lower():
                common_match = common_name
                break

        if common_match:
            self._common_name = common_match
            self._latin_name = common_to_latin[common_match]
            return

        # If we get here, the species wasn't found
        available_species = list(LATIN_TO_COMMON_SPECIES_NAMES.keys()) + list(
            common_to_latin.keys()
        )
        raise ValueError(
            f"Unknown species: '{species_input}'. Available species: {available_species}"
        )

    @property
    def latin_name(self) -> str:
        """
        Get the Latin species name.

        Returns
        -------
        str
            The Latin species name (e.g., 'Homo sapiens').
        """
        return self._latin_name

    @property
    def common_name(self) -> str:
        """
        Get the common species name.

        Returns
        -------
        str
            The common species name (e.g., 'human').
        """
        return self._common_name

    def __str__(self) -> str:
        """
        Return a human-readable string representation.

        Returns
        -------
        str
            String in format 'common_name (Latin_name)'.
        """
        return f"{self.common_name} ({self.latin_name})"

    def __repr__(self) -> str:
        """
        Return a detailed string representation for debugging.

        Returns
        -------
        str
            String representation of the SpeciesValidator instance.
        """
        return f"SpeciesValidator('{self.latin_name}')"

    @classmethod
    def get_available_species(cls) -> Dict[str, list]:
        """
        Return a dictionary of all available species names.

        Returns
        -------
        Dict[str, list]
            Dictionary with keys 'latin_names' and 'common_names', each
            containing a list of available species names.

        Examples
        --------
        >>> available = SpeciesValidator.get_available_species()
        >>> available['latin_names']
        ['Homo sapiens', 'Mus musculus', ...]
        >>> available['common_names']
        ['human', 'mouse', ...]
        """
        common_to_latin = {
            common: latin for latin, common in LATIN_TO_COMMON_SPECIES_NAMES.items()
        }
        return {
            "latin_names": list(LATIN_TO_COMMON_SPECIES_NAMES.keys()),
            "common_names": list(common_to_latin.keys()),
        }

    def validate_against_supported(
        self, supported_species: Union[List[str], Set[str]]
    ) -> bool:
        """
        Validate that this species is supported by a specific function or analysis.

        Parameters
        ----------
        supported_species : Union[List[str], Set[str]]
            Collection of supported species names. Can contain either common names,
            Latin names, or a mix of both. Case-insensitive matching.

        Returns
        -------
        bool
            True if this species is in the supported list, False otherwise.

        Examples
        --------
        >>> species = SpeciesValidator("human")
        >>> species.validate_against_supported(["human", "mouse", "rat"])
        True

        >>> species = SpeciesValidator("Homo sapiens")
        >>> species.validate_against_supported(["Homo sapiens", "Mus musculus"])
        True

        >>> species = SpeciesValidator("fly")
        >>> species.validate_against_supported(["human", "mouse"])
        False
        """
        # Normalize supported species (case-insensitive)
        normalized_supported = {
            species.strip().lower() for species in supported_species
        }

        # Check if either our common name or Latin name is in the supported list
        return (
            self.common_name.lower() in normalized_supported
            or self.latin_name.lower() in normalized_supported
        )

    def assert_supported(
        self, supported_species: Union[List[str], Set[str]], context: str = ""
    ) -> None:
        """
        Assert that this species is supported, raising an exception if not.

        Parameters
        ----------
        supported_species : Union[List[str], Set[str]]
            Collection of supported species names. Can contain either common names,
            Latin names, or a mix of both. Case-insensitive matching.
        context : str, optional
            Additional context for the error message (e.g., function name).

        Raises
        ------
        ValueError
            If this species is not in the supported list.

        Examples
        --------
        >>> species = SpeciesValidator("human")
        >>> species.assert_supported(["human", "mouse"], "my_analysis_function")
        # No exception raised

        >>> species = SpeciesValidator("fly")
        >>> species.assert_supported(["human", "mouse"], "my_analysis_function")
        ValueError: Species 'fly (Drosophila melanogaster)' not supported by my_analysis_function...
        """
        if not self.validate_against_supported(supported_species):
            context_str = f" by {context}" if context else ""
            supported_display = ", ".join(f"'{s}'" for s in supported_species)
            raise ValueError(
                f"Species '{self}' not supported{context_str}. "
                f"Supported species: {supported_display}"
            )

    def lookup_custom_value(
        self, custom_table: Dict[str, Any], is_latin: bool = True
    ) -> Any:
        """
        Look up a custom value for this species from a provided table.

        Parameters
        ----------
        custom_table : Dict[str, Any]
            Dictionary mapping species names to custom values. Keys should be
            either Latin names (if is_latin=True) or common names (if is_latin=False).
        is_latin : bool, default True
            If True, treats the keys in custom_table as Latin species names.
            If False, treats the keys as common species names.

        Returns
        -------
        Any
            The value associated with this species in the custom table.

        Raises
        ------
        ValueError
            If this species is not found in the custom table.

        Examples
        --------
        >>> # Custom table with Latin names as keys
        >>> psi_table = {
        ...     "Homo sapiens": "human",
        ...     "Mus musculus": "mouse",
        ...     "Saccharomyces cerevisiae": "yeast"
        ... }
        >>> species = SpeciesValidator("human")
        >>> species.lookup_custom_value(psi_table, is_latin=True)
        'human'

        >>> # Custom table with common names as keys
        >>> custom_ids = {
        ...     "human": "HUMAN_001",
        ...     "mouse": "MOUSE_001"
        ... }
        >>> species = SpeciesValidator("Homo sapiens")
        >>> species.lookup_custom_value(custom_ids, is_latin=False)
        'HUMAN_001'

        >>> # Species not in table raises error
        >>> species = SpeciesValidator("fly")
        >>> species.lookup_custom_value(psi_table, is_latin=True)
        ValueError: Species 'fly (Drosophila melanogaster)' not found in custom table...
        """
        # Determine which of our names to use for lookup
        lookup_key = self.latin_name if is_latin else self.common_name

        # Case-insensitive lookup
        for table_key, value in custom_table.items():
            if table_key.lower() == lookup_key.lower():
                return value

        # If we get here, the species wasn't found
        key_type = "Latin names" if is_latin else "common names"
        available_keys = list(custom_table.keys())
        raise ValueError(
            f"Species '{self}' not found in custom table. "
            f"Table contains {key_type}: {available_keys}. "
            f"Looking for: '{lookup_key}'"
        )
