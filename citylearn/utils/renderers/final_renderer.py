from citylearn.utils.renderers.renderer import Renderer
from datetime import date
from collections import defaultdict
from pandas import DataFrame

class FinalRenderer(Renderer):

    _defer_flush : bool
    _buffer : dict[str, list[list]]

    data = [
        {'Net Electricity Consumption-kWh': 16.537399291992188, 'Non-shiftable Load-kWh': 2.2757999897003174, 'Non-shiftable Load Electricity Consumption-kWh': 6.827400207519531, 'Energy Production from PV-kWh': -0.0},
        {'Net Electricity Consumption-kWh': 8.863166809082031, 'Non-shiftable Load-kWh': 0.8511666655540466, 'Non-shiftable Load Electricity Consumption-kWh': 0.8511666655540466, 'Energy Production from PV-kWh': -0.0},
        {'Net Electricity Consumption-kWh': 5.689599514007568, 'Non-shiftable Load-kWh': 0.8345999717712402, 'Non-shiftable Load Electricity Consumption-kWh': 0.8345999717712402, 'Energy Production from PV-kWh': -0.0},
        {'Net Electricity Consumption-kWh': 5.693166255950928, 'Non-shiftable Load-kWh': 0.8381666541099548, 'Non-shiftable Load Electricity Consumption-kWh': 0.8381666541099548, 'Energy Production from PV-kWh': -0.0},
        {'Net Electricity Consumption-kWh': 2.8212435245513916, 'Non-shiftable Load-kWh': 1.47843337059021, 'Non-shiftable Load Electricity Consumption-kWh': 1.47843337059021, 'Energy Production from PV-kWh': -0.0},
        {'Net Electricity Consumption-kWh': 1.882153034210205, 'Non-shiftable Load-kWh': 1.2561999559402466, 'Non-shiftable Load Electricity Consumption-kWh': 1.2561999559402466, 'Energy Production from PV-kWh': -0.0},
        {'Net Electricity Consumption-kWh': 2.2381997108459473, 'Non-shiftable Load-kWh': 1.8695166110992432, 'Non-shiftable Load Electricity Consumption-kWh': 1.8695166110992432, 'Energy Production from PV-kWh': -0.17404999923706055},
        {'Net Electricity Consumption-kWh': 0.0036388293374329805, 'Non-shiftable Load-kWh': 0.80881667137146, 'Non-shiftable Load Electricity Consumption-kWh': 0.80881667137146, 'Energy Production from PV-kWh': -1.3382500305175782},
        {'Net Electricity Consumption-kWh': -2.1706833839416504, 'Non-shiftable Load-kWh': 0.6159666776657104, 'Non-shiftable Load Electricity Consumption-kWh': 0.6159666776657104, 'Energy Production from PV-kWh': -3.31764990234375},
        {'Net Electricity Consumption-kWh': -5.133281707763672, 'Non-shiftable Load-kWh': 0.6272333264350891, 'Non-shiftable Load Electricity Consumption-kWh': 0.6272333264350891, 'Energy Production from PV-kWh': -5.436299926757813},
        {'Net Electricity Consumption-kWh': -14.265483856201172, 'Non-shiftable Load-kWh': 0.618066668510437, 'Non-shiftable Load Electricity Consumption-kWh': 0.618066668510437, 'Energy Production from PV-kWh': -7.28355029296875},
        {'Net Electricity Consumption-kWh': -15.507866859436035, 'Non-shiftable Load-kWh': 0.6451333165168762, 'Non-shiftable Load Electricity Consumption-kWh': 0.6451333165168762, 'Energy Production from PV-kWh': -8.553},
        {'Net Electricity Consumption-kWh': -15.95346736907959, 'Non-shiftable Load-kWh': 0.7644333243370056, 'Non-shiftable Load Electricity Consumption-kWh': 0.7644333243370056, 'Energy Production from PV-kWh': -9.117900146484375},
        {'Net Electricity Consumption-kWh': -7.982850074768066, 'Non-shiftable Load-kWh': 1.4322999715805054, 'Non-shiftable Load Electricity Consumption-kWh': 1.4322999715805054, 'Energy Production from PV-kWh': -9.015150146484375},
        {'Net Electricity Consumption-kWh': -6.774266719818115, 'Non-shiftable Load-kWh': 1.9018332958221436, 'Non-shiftable Load Electricity Consumption-kWh': 1.9018332958221436, 'Energy Production from PV-kWh': -8.276099853515625},
        {'Net Electricity Consumption-kWh': -5.645317077636719, 'Non-shiftable Load-kWh': 1.7299833297729492, 'Non-shiftable Load Electricity Consumption-kWh': 1.7299833297729492, 'Energy Production from PV-kWh': -6.97530029296875},
        {'Net Electricity Consumption-kWh': -4.077666282653809, 'Non-shiftable Load-kWh': 1.3914333581924438, 'Non-shiftable Load Electricity Consumption-kWh': 1.3914333581924438, 'Energy Production from PV-kWh': -5.069099853515625},
        {'Net Electricity Consumption-kWh': -6.291865825653076, 'Non-shiftable Load-kWh': 1.0300999879837036, 'Non-shiftable Load Electricity Consumption-kWh': 1.0300999879837036, 'Energy Production from PV-kWh': -2.9112000732421874},
        {'Net Electricity Consumption-kWh': -3.923083543777466, 'Non-shiftable Load-kWh': 1.3838167190551758, 'Non-shiftable Load Electricity Consumption-kWh': 1.3838167190551758, 'Energy Production from PV-kWh': -0.9869000244140625},
        {'Net Electricity Consumption-kWh': -3.3628835678100586, 'Non-shiftable Load-kWh': 1.031916618347168, 'Non-shiftable Load Electricity Consumption-kWh': 1.031916618347168, 'Energy Production from PV-kWh': -0.07479999732971192},
        {'Net Electricity Consumption-kWh': 12.403982162475586, 'Non-shiftable Load-kWh': 3.6039834022521973, 'Non-shiftable Load Electricity Consumption-kWh': 3.6039834022521973, 'Energy Production from PV-kWh': -0.0},
        {'Net Electricity Consumption-kWh': 13.808499336242676, 'Non-shiftable Load-kWh': 5.008500099182129, 'Non-shiftable Load Electricity Consumption-kWh': 5.008500099182129, 'Energy Production from PV-kWh': -0.0},
        {'Net Electricity Consumption-kWh': 13.151216506958008, 'Non-shiftable Load-kWh': 3.896216630935669, 'Non-shiftable Load Electricity Consumption-kWh': 3.896216630935669, 'Energy Production from PV-kWh': -0.0},
        {'Net Electricity Consumption-kWh': 12.173139572143555, 'Non-shiftable Load-kWh': 3.5570833683013916, 'Non-shiftable Load Electricity Consumption-kWh': 3.5570833683013916, 'Energy Production from PV-kWh': -0.0},
        {'Net Electricity Consumption-kWh': 2.81278395652771, 'Non-shiftable Load-kWh': 1.4113333225250244, 'Non-shiftable Load Electricity Consumption-kWh': 1.4113333225250244, 'Energy Production from PV-kWh': -0.0},
        {'Net Electricity Consumption-kWh': 1.6041923761367798, 'Non-shiftable Load-kWh': 0.9794166684150696, 'Non-shiftable Load Electricity Consumption-kWh': 0.9794166684150696, 'Energy Production from PV-kWh': -0.0},
        {'Net Electricity Consumption-kWh': 1.4435688257217407, 'Non-shiftable Load-kWh': 0.9009749889373779, 'Non-shiftable Load Electricity Consumption-kWh': 0.9009749889373779, 'Energy Production from PV-kWh': -0.0},
        {'Net Electricity Consumption-kWh': 1.4554862976074219, 'Non-shiftable Load-kWh': 0.9224333167076111, 'Non-shiftable Load Electricity Consumption-kWh': 0.9224333167076111, 'Energy Production from PV-kWh': -0.0},
        {'Net Electricity Consumption-kWh': 1.4424303770065308, 'Non-shiftable Load-kWh': 0.9104833602905273, 'Non-shiftable Load Electricity Consumption-kWh': 0.9104833602905273, 'Energy Production from PV-kWh': -0.0},
        {'Net Electricity Consumption-kWh': 11.88711929321289, 'Non-shiftable Load-kWh': 1.0103000402450562, 'Non-shiftable Load Electricity Consumption-kWh': 1.0103000402450562, 'Energy Production from PV-kWh': -0.0},
        {'Net Electricity Consumption-kWh': 1.6603366136550903, 'Non-shiftable Load-kWh': 1.2976833581924438, 'Non-shiftable Load Electricity Consumption-kWh': 1.2976833581924438, 'Energy Production from PV-kWh': -0.16915000534057617},
        {'Net Electricity Consumption-kWh': 0.40861958265304565, 'Non-shiftable Load-kWh': 1.1922667026519775, 'Non-shiftable Load Electricity Consumption-kWh': 1.1922667026519775, 'Energy Production from PV-kWh': -1.3154500122070312},
        {'Net Electricity Consumption-kWh': -1.4787671566009521, 'Non-shiftable Load-kWh': 1.2234666347503662, 'Non-shiftable Load Electricity Consumption-kWh': 1.2234666347503662, 'Energy Production from PV-kWh': -3.233099853515625},
        {'Net Electricity Consumption-kWh': -4.604245185852051, 'Non-shiftable Load-kWh': 1.0826833248138428, 'Non-shiftable Load Electricity Consumption-kWh': 1.0826833248138428, 'Energy Production from PV-kWh': -5.362699951171875},
        {'Net Electricity Consumption-kWh': -14.060733795166016, 'Non-shiftable Load-kWh': 0.6676666736602783, 'Non-shiftable Load Electricity Consumption-kWh': 0.6676666736602783, 'Energy Production from PV-kWh': -7.12839990234375},
        {'Net Electricity Consumption-kWh': -15.313650131225586, 'Non-shiftable Load-kWh': 0.666100025177002, 'Non-shiftable Load Electricity Consumption-kWh': 0.666100025177002, 'Energy Production from PV-kWh': -8.37975},
        {'Net Electricity Consumption-kWh': -15.966716766357422, 'Non-shiftable Load-kWh': 0.6456833481788635, 'Non-shiftable Load Electricity Consumption-kWh': 0.6456833481788635, 'Energy Production from PV-kWh': -9.01239990234375},
        {'Net Electricity Consumption-kWh': -8.098382949829102, 'Non-shiftable Load-kWh': 1.1646167039871216, 'Non-shiftable Load Electricity Consumption-kWh': 1.1646167039871216, 'Energy Production from PV-kWh': -8.862999755859375},
        {'Net Electricity Consumption-kWh': -7.1304168701171875, 'Non-shiftable Load-kWh': 1.443583369255066, 'Non-shiftable Load Electricity Consumption-kWh': 1.443583369255066, 'Energy Production from PV-kWh': -8.174000244140625},
        {'Net Electricity Consumption-kWh': -6.392800331115723, 'Non-shiftable Load-kWh': 0.8619499802589417, 'Non-shiftable Load Electricity Consumption-kWh': 0.8619499802589417, 'Energy Production from PV-kWh': -6.854750244140625},
        {'Net Electricity Consumption-kWh': -4.672500133514404, 'Non-shiftable Load-kWh': 0.7617499828338623, 'Non-shiftable Load Electricity Consumption-kWh': 0.7617499828338623, 'Energy Production from PV-kWh': -5.034250122070312},
        {'Net Electricity Consumption-kWh': -0.37523353099823, 'Non-shiftable Load-kWh': 2.9404666423797607, 'Non-shiftable Load Electricity Consumption-kWh': 2.9404666423797607, 'Energy Production from PV-kWh': -2.9157000732421876},
        {'Net Electricity Consumption-kWh': 1.0572665929794312, 'Non-shiftable Load-kWh': 2.4557666778564453, 'Non-shiftable Load Electricity Consumption-kWh': 2.4557666778564453, 'Energy Production from PV-kWh': -0.9985000305175781},
        {'Net Electricity Consumption-kWh': 0.8391194343566895, 'Non-shiftable Load-kWh': 5.264016628265381, 'Non-shiftable Load Electricity Consumption-kWh': 5.264016628265381, 'Energy Production from PV-kWh': -0.08285000038146972},
        {'Net Electricity Consumption-kWh': 13.99293327331543, 'Non-shiftable Load-kWh': 5.192933559417725, 'Non-shiftable Load Electricity Consumption-kWh': 5.192933559417725, 'Energy Production from PV-kWh': -0.0},
        {'Net Electricity Consumption-kWh': 11.764150619506836, 'Non-shiftable Load-kWh': 3.9318833351135254, 'Non-shiftable Load Electricity Consumption-kWh': 3.9318833351135254, 'Energy Production from PV-kWh': -0.0},
        {'Net Electricity Consumption-kWh': 3.260903835296631, 'Non-shiftable Load-kWh': 1.8948500156402588, 'Non-shiftable Load Electricity Consumption-kWh': 1.8948500156402588, 'Energy Production from PV-kWh': -0.0},
        {'Net Electricity Consumption-kWh': 1.9743298292160034, 'Non-shiftable Load-kWh': 1.3533666133880615, 'Non-shiftable Load Electricity Consumption-kWh': 1.3533666133880615, 'Energy Production from PV-kWh': -0.0},
        {'Net Electricity Consumption-kWh': 2.3312017917633057, 'Non-shiftable Load-kWh': 1.7904499769210815, 'Non-shiftable Load Electricity Consumption-kWh': 1.7904499769210815, 'Energy Production from PV-kWh': -0.0},
        {'Net Electricity Consumption-kWh': 20.279449462890625, 'Non-shiftable Load-kWh': 0.9643333554267883, 'Non-shiftable Load Electricity Consumption-kWh': 0.9643333554267883, 'Energy Production from PV-kWh': -0.0},
        {'Net Electricity Consumption-kWh': 1.501621961593628, 'Non-shiftable Load-kWh': 0.9697833061218262, 'Non-shiftable Load Electricity Consumption-kWh': 0.9697833061218262, 'Energy Production from PV-kWh': -0.0},
        {'Net Electricity Consumption-kWh': 1.5118376016616821, 'Non-shiftable Load-kWh': 0.9800333380699158, 'Non-shiftable Load Electricity Consumption-kWh': 0.9800333380699158, 'Energy Production from PV-kWh': -0.0},
        {'Net Electricity Consumption-kWh': 1.5054835081100464, 'Non-shiftable Load-kWh': 0.9736833572387695, 'Non-shiftable Load Electricity Consumption-kWh': 0.9736833572387695, 'Energy Production from PV-kWh': -0.0},
        {'Net Electricity Consumption-kWh': 1.9506497383117676, 'Non-shiftable Load-kWh': 1.4188499450683594, 'Non-shiftable Load Electricity Consumption-kWh': 1.4188499450683594, 'Energy Production from PV-kWh': -0.0},
        {'Net Electricity Consumption-kWh': 2.1509697437286377, 'Non-shiftable Load-kWh': 1.8283666372299194, 'Non-shiftable Load Electricity Consumption-kWh': 1.8283666372299194, 'Energy Production from PV-kWh': -0.20919998931884765},
        {'Net Electricity Consumption-kWh': 0.5472193956375122, 'Non-shiftable Load-kWh': 1.4670166969299316, 'Non-shiftable Load Electricity Consumption-kWh': 1.4670166969299316, 'Energy Production from PV-kWh': -1.4516000061035157},
        {'Net Electricity Consumption-kWh': -1.7691835165023804, 'Non-shiftable Load-kWh': 0.7758499975204468, 'Non-shiftable Load Electricity Consumption-kWh': 0.7758499975204468, 'Energy Production from PV-kWh': -3.24598388671875},
        {'Net Electricity Consumption-kWh': -4.832433700561523, 'Non-shiftable Load-kWh': 0.697416663646698, 'Non-shiftable Load Electricity Consumption-kWh': 0.697416663646698, 'Energy Production from PV-kWh': -5.529983520507813},
        {'Net Electricity Consumption-kWh': -14.064350128173828, 'Non-shiftable Load-kWh': 0.6469333176612854, 'Non-shiftable Load Electricity Consumption-kWh': 0.6469333176612854, 'Energy Production from PV-kWh': -7.2041168212890625},
        {'Net Electricity Consumption-kWh': -15.338300704956055, 'Non-shiftable Load-kWh': 0.6526333093643188, 'Non-shiftable Load Electricity Consumption-kWh': 0.6526333093643188, 'Energy Production from PV-kWh': -8.594366455078125},
        {'Net Electricity Consumption-kWh': -15.894366264343262, 'Non-shiftable Load-kWh': 0.6449666628837585, 'Non-shiftable Load Electricity Consumption-kWh': 0.6449666628837585, 'Energy Production from PV-kWh': -9.1895},
        {'Net Electricity Consumption-kWh': -8.060200691223145, 'Non-shiftable Load-kWh': 1.407200008392334, 'Non-shiftable Load Electricity Consumption-kWh': 1.407200008392334, 'Energy Production from PV-kWh': -9.116500244140625},
        {'Net Electricity Consumption-kWh': -7.169450283050537, 'Non-shiftable Load-kWh': 1.8701166515350342, 'Non-shiftable Load Electricity Consumption-kWh': 1.8701166515350342, 'Energy Production from PV-kWh': -8.149933471679688},
        {'Net Electricity Consumption-kWh': -6.397566795349121, 'Non-shiftable Load-kWh': 1.3741333484649658, 'Non-shiftable Load Electricity Consumption-kWh': 1.3741333484649658, 'Energy Production from PV-kWh': -6.4488330078125},
        {'Net Electricity Consumption-kWh': -4.741016864776611, 'Non-shiftable Load-kWh': 1.2886000270843506, 'Non-shiftable Load Electricity Consumption-kWh': 1.2886000270843506, 'Energy Production from PV-kWh': -4.52885009765625}
    ]

    def __init__(self, directory: str, flag : str, session_name : str,
                start_date : date, defer_flush = True, enabled = True):
        
        super().__init__(directory, flag, session_name,
                        start_date, enabled)
        
        self._defer_flush = defer_flush
        self._buffer = defaultdict(list)

    def export_csv(self, filename : str, data : dict):

        if self._defer_flush:
            current_table = self._buffer.get(filename)
            for current_column, new_observation in zip(current_table, data.values()):
                current_column.append(new_observation)
            return

        complete_data = self._buffer.get(filename)
        pandas_table = DataFrame(complete_data)
        pandas_table.to_csv(f"{self._directory}/{filename}" ,index=False)
        
        self._flush_render_buffer()
        self._defer_flush = True

    def _flush_render_buffer(self):
        pass
            



render = FinalRenderer("here/ola", "", "ola", None, None)
