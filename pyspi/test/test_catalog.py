from pyspi import SPICatalog, pyspi_config


def test_catalog():

    spi = SPICatalog()

    spi.query_object(
        "Jupiter",
        radius=10.0 * u.deg,
        start_date="2021-01-01",
        end_date="2022-02-15",
    )

    spi.query_object(
        "Jupiter",
        radius=10.0 * u.deg,
        start_date="2021-01-01",
        end_date="2022-02-15",
        query="SCW_TYPE=='POINTING'",
    )

    assert len(spi.current_table) == 2

    my_windows = spi.science_windows.get_filtered_set(
        pointing_type="pointing", public=True
    )

    assert len(my_windows) == 1

    my_windows.get()

    my_windows.files()
