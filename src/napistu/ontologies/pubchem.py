import logging
import time
from typing import Any, Dict, List, Tuple

import requests

from napistu.ontologies.constants import (
    PUBCHEM_DEFS,
    PUBCHEM_ID_ENTRYPOINT,
    PUBCHEM_PROPERTIES,
)

# Setup logger
logger = logging.getLogger(__name__)


class PubChemConnectivityError(Exception):
    """Raised when PubChem API is unreachable due to network/connectivity issues."""

    pass


def map_pubchem_ids(
    pubchem_cids: List[str],
    batch_size: int = 100,
    max_retries: int = 3,
    delay: float = 0.25,
    verbose: bool = True,
) -> Dict[str, Dict[str, str]]:
    """
    Map PubChem Compound Identifiers (CIDs) to compound names and SMILES strings.

    Efficiently processes large datasets using batched API requests with retry logic.
    Returns both compound names and Isomeric SMILES for each CID.

    Parameters
    ----------
    pubchem_cids : List[str]
        List of PubChem CIDs as strings (e.g., ["2244", "5362065"]).
    batch_size : int, optional
        CIDs per API request. Default 100. Range: 1-500.
    max_retries : int, optional
        Retry attempts per failed batch. Default 3.
    delay : float, optional
        Seconds between requests. Default 0.25 (respects 5 req/sec limit).
    verbose : bool, optional
        Enable detailed logging. Default True.

    Returns
    -------
    Dict[str, Dict[str, str]]
        Maps CID to {"name": str, "smiles": str}. Missing data returns empty strings.

    Examples
    --------
    >>> result = map_pubchem_ids(["2244", "5362065"])
    >>> print(result["2244"])
    {"name": "aspirin", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"}

    Notes
    -----
    - Rate limit: Max 5 requests/second (PubChem policy)
    - Timeout: 30 seconds per request
    - Uses Isomeric SMILES (includes stereochemistry)
    - Some compounds may lack SMILES data
    """
    # Set logging level
    logger.setLevel(logging.INFO if verbose else logging.WARNING)

    # Validate inputs
    if not pubchem_cids:
        logger.warning("Empty CID list provided")
        return {}

    batch_size, max_retries, delay = _validate_params(batch_size, max_retries, delay)
    num_batches = (len(pubchem_cids) + batch_size - 1) // batch_size

    if verbose:
        logger.info(f"Mapping {len(pubchem_cids)} CIDs in {num_batches} batches")
        logger.info(
            f"Rate: {1/delay:.1f} req/sec, Est. time: {num_batches * delay / 60:.1f} min"
        )

    # Process batches
    results = {}
    failed_cids = []

    for i in range(0, len(pubchem_cids), batch_size):
        batch = pubchem_cids[i : i + batch_size]
        batch_num = i // batch_size + 1

        if verbose:
            logger.info(
                f"Processing batch {batch_num}/{num_batches} ({len(batch)} CIDs)"
            )

        batch_results, success = _fetch_batch(batch, max_retries, delay)

        if success:
            results.update(batch_results)
            # Log batch stats
            if verbose:
                found = sum(
                    1
                    for data in batch_results.values()
                    if data[PUBCHEM_DEFS.NAME] != data[PUBCHEM_DEFS.NAME]
                    or data[PUBCHEM_DEFS.NAME]
                    in [
                        v[PUBCHEM_DEFS.NAME]
                        for v in batch_results.values()
                        if v[PUBCHEM_DEFS.NAME]
                        != list(batch_results.keys())[
                            list(batch_results.values()).index(v)
                        ]
                    ]
                )
                with_smiles = sum(
                    1 for data in batch_results.values() if data[PUBCHEM_DEFS.SMILES]
                )
                # Simpler count: just check if name != CID
                found = sum(
                    1
                    for cid, data in batch_results.items()
                    if data[PUBCHEM_DEFS.NAME] != cid
                )
                logger.info(
                    f"  ✓ Found {found}/{len(batch)} compounds, {with_smiles} with SMILES"
                )
            time.sleep(delay)
        else:
            failed_cids.extend(batch)

    # Handle completely failed CIDs
    for cid in failed_cids:
        results[cid] = {PUBCHEM_DEFS.NAME: cid, PUBCHEM_DEFS.SMILES: ""}

    # Final stats
    successful = sum(
        1 for cid, data in results.items() if data[PUBCHEM_DEFS.NAME] != cid
    )
    with_smiles = sum(1 for data in results.values() if data[PUBCHEM_DEFS.SMILES])

    log_msg = (
        f"Complete: {successful}/{len(pubchem_cids)} mapped, {with_smiles} with SMILES"
    )
    if verbose:
        logger.info(log_msg)
    elif successful < len(pubchem_cids):
        logger.warning(log_msg)

    return results


def _validate_params(
    batch_size: int, max_retries: int, delay: float
) -> Tuple[int, int, float]:
    """Validate and correct input parameters."""
    if batch_size <= 0 or batch_size > 500:
        logger.warning(f"Invalid batch_size {batch_size}, using default 100")
        batch_size = 100

    if max_retries < 0:
        logger.warning(f"Invalid max_retries {max_retries}, using 0")
        max_retries = 0

    if delay < 0:
        logger.warning(f"Invalid delay {delay}, using 0")
        delay = 0

    return batch_size, max_retries, delay


def _process_batch_response(
    data: Dict[str, Any], batch: List[str]
) -> Dict[str, Dict[str, str]]:
    """
    Process PubChem API response for a batch of compound identifiers.

    Extracts compound names and SMILES strings from the PubChem REST API response,
    handling both found and missing compounds in the batch.

    Parameters
    ----------
    data : Dict[str, any]
        JSON response from PubChem REST API containing property table with
        compound information.
    batch : List[str]
        List of PubChem CIDs that were requested in this batch.

    Returns
    -------
    Dict[str, Dict[str, str]]
        Dictionary mapping each CID to a nested dictionary containing:
        - 'name': Compound name (prefers Title over IUPACName, falls back to CID)
        - 'smiles': Isomeric SMILES string (empty string if not available)

        Missing CIDs are included with name=CID and empty SMILES.

    Notes
    -----
    - Handles API inconsistencies between IsomericSMILES and SMILES fields
    - Compounds not found in API response are included with default values
    - Prefers compound Title over IUPACName for better readability
    """
    batch_results = {}
    properties_list = data.get(PUBCHEM_PROPERTIES.PROPERTY_TABLE, {}).get(
        PUBCHEM_PROPERTIES.PROPERTIES, []
    )
    found_cids = set()

    # Process found compounds
    for prop in properties_list:
        cid = str(prop.get(PUBCHEM_PROPERTIES.CID))
        found_cids.add(cid)

        # Prefer Title over IUPACName
        name = prop.get(
            PUBCHEM_PROPERTIES.TITLE, prop.get(PUBCHEM_PROPERTIES.IUPAC_NAME, cid)
        )

        # Try both SMILES field names (API seems inconsistent)
        smiles = prop.get(
            PUBCHEM_PROPERTIES.ISOMERIC_SMILES, prop.get(PUBCHEM_PROPERTIES.SMILES, "")
        )

        batch_results[cid] = {PUBCHEM_DEFS.NAME: name, PUBCHEM_DEFS.SMILES: smiles}

    # Handle missing CIDs
    for cid in batch:
        if cid not in found_cids:
            batch_results[cid] = {PUBCHEM_DEFS.NAME: cid, PUBCHEM_DEFS.SMILES: ""}

    return batch_results


def _fetch_batch(
    batch: List[str], max_retries: int, delay: float
) -> Tuple[Dict[str, Dict[str, str]], bool]:
    """Fetch data for a single batch of CIDs with retry logic."""
    cids_str = ",".join(batch)
    # Request both SMILES formats to handle API inconsistencies
    url = PUBCHEM_ID_ENTRYPOINT.format(cids_str=cids_str)

    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, timeout=60)

            if response.status_code == 200:
                data = response.json()
                batch_results = _process_batch_response(data, batch)
                return batch_results, True

            elif response.status_code == 400:
                # Invalid CIDs - don't retry but try individual requests for mixed batches
                if len(batch) > 1:
                    logger.warning(
                        "Batch with mixed CIDs failed, trying individual requests"
                    )
                    return _fetch_individual_cids(batch, max_retries, delay)
                else:
                    logger.warning(f"Invalid CID: {batch[0]}")
                    return {
                        batch[0]: {PUBCHEM_DEFS.NAME: batch[0], PUBCHEM_DEFS.SMILES: ""}
                    }, True

            else:
                if attempt == max_retries:
                    logger.error(
                        f"API error ({response.status_code}) after {max_retries + 1} attempts"
                    )
                else:
                    logger.warning(
                        f"Retry {attempt + 1}/{max_retries + 1} (HTTP {response.status_code})"
                    )
                    time.sleep(delay * 2)

        except Exception as e:
            # Immediate dealbreakers - re-raise right away
            if _is_immediate_failure(e):
                raise PubChemConnectivityError(f"Network error: {e}") from e

            # Potentially transient issues - retry logic
            if attempt == max_retries:
                # Failed after all retries - now raise
                raise PubChemConnectivityError(
                    f"API error after {max_retries + 1} attempts: {e}"
                ) from e
            else:
                logger.warning(f"Retry {attempt + 1}/{max_retries + 1} - {e}")
                time.sleep(delay * 2)

    return {}, False


def _fetch_individual_cids(
    cids: List[str], max_retries: int, delay: float
) -> Tuple[Dict[str, Dict[str, str]], bool]:
    """Fetch CIDs individually when batch fails due to mixed valid/invalid CIDs."""
    results = {}

    for cid in cids:
        individual_result, success = _fetch_batch([cid], max_retries, delay)
        if success:
            results.update(individual_result)
        else:
            results[cid] = {PUBCHEM_DEFS.NAME: cid, PUBCHEM_DEFS.SMILES: ""}
        time.sleep(delay * 0.5)  # Shorter delay for individual requests

    return results, True


def _is_immediate_failure(e):
    """Return True if this error should not be retried."""
    immediate_failures = (
        requests.exceptions.SSLError,
        requests.exceptions.ConnectionError,  # DNS, network unreachable
        # Could add others like specific SSL cert errors
    )
    return isinstance(e, immediate_failures)
