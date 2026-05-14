import asyncio

from agentic_unitree_go2.robot.mock import MockConnection, is_mock_conn


RTC_TOPIC = {
    "LF_SPORT_MOD_STATE": "sport",
    "LOW_STATE": "low",
    "MULTIPLE_STATE": "multi",
}


def test_mock_connection_emits_state_updates():
    async def run():
        conn = MockConnection(RTC_TOPIC)
        sport_updates = []
        low_updates = []

        conn.datachannel.pub_sub.subscribe(RTC_TOPIC["LF_SPORT_MOD_STATE"], sport_updates.append)
        conn.datachannel.pub_sub.subscribe(RTC_TOPIC["LOW_STATE"], low_updates.append)
        await conn.datachannel.pub_sub.publish_request_new("topic", {"parameter": {"x": 1}})

        assert is_mock_conn(conn)
        assert sport_updates[-1]["data"]["velocity"][0] == 1.0
        assert low_updates[-1]["data"]["bms_state"]["soc"] == 87

    asyncio.run(run())
