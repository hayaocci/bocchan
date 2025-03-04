// PIN定義：LED(プログラム状態)
#define LED_RUN PIN_LED0
// PIN定義：LED(GNSS電波状態)
#define LED_GNSS PIN_LED1
// PIN定義：LED(ELTRES状態)
#define LED_SND PIN_LED2
// PIN定義：LED(エラー状態)
#define LED_ERR PIN_LED3

// プログラム内部状態：初期状態
#define PROGRAM_STS_INIT (0)
// プログラム内部状態：起動中
#define PROGRAM_STS_RUNNING (1)
// プログラム内部状態：終了
#define PROGRAM_STS_STOPPED (3)

// プログラム内部状態
int program_sts = PROGRAM_STS_INIT;
// GNSS電波受信タイムアウト（GNSS受信エラー）発生フラグ
bool gnss_recevie_timeout = false;
// 点滅処理で最後に変更した時間
uint64_t last_change_blink_time = 0;
// イベント通知での送信直前通知（5秒前）受信フラグ
bool event_send_ready = false;
// ペイロードデータ格納場所
uint8_t payload[16];
// 最新のGGA情報
eltres_board_gga_info last_gga_info;

// int num_people = 23;

/**
 * @brief イベント通知受信コールバック
 * @param event イベント種別
 */
void eltres_event_cb(eltres_board_event event)
{
    switch (event)
    {
    case ELTRES_BOARD_EVT_GNSS_TMOUT:
        // GNSS電波受信タイムアウト
        Serial.println("gnss wait timeout error.");
        gnss_recevie_timeout = true;
        break;
    case ELTRES_BOARD_EVT_IDLE:
        // アイドル状態
        Serial.println("waiting sending timings.");
        digitalWrite(LED_SND, LOW);
        break;
    case ELTRES_BOARD_EVT_SEND_READY:
        // 送信直前通知（5秒前）
        Serial.println("Shortly before sending, so setup payload if need.");
        event_send_ready = true;
        break;
    case ELTRES_BOARD_EVT_SENDING:
        // 送信開始
        Serial.println("start sending.");
        digitalWrite(LED_SND, HIGH);
        break;
    case ELTRES_BOARD_EVT_GNSS_UNRECEIVE:
        // GNSS電波未受信
        Serial.println("gnss wave has not been received.");
        digitalWrite(LED_GNSS, LOW);
        break;
    case ELTRES_BOARD_EVT_GNSS_RECEIVE:
        // GNSS電波受信
        Serial.println("gnss wave has been received.");
        digitalWrite(LED_GNSS, HIGH);
        gnss_recevie_timeout = false;
        break;
    case ELTRES_BOARD_EVT_FAULT:
        // 内部エラー発生
        Serial.println("internal error.");
        break;
    }
}
/**
 * @brief GGA情報受信コールバック
 * @param gga_info GGA情報のポインタ
 */
void gga_event_cb(const eltres_board_gga_info *gga_info)
{
    Serial.print("[gga]");
    last_gga_info = *gga_info;
    if (gga_info->m_pos_status)
    {
        // 測位状態
        // GGA情報をシリアルモニタへ出力
        Serial.print("utc: ");
        Serial.println((const char *)gga_info->m_utc);
        Serial.print("lat: ");
        Serial.print((const char *)gga_info->m_n_s);
        Serial.print((const char *)gga_info->m_lat);
        Serial.print(", lon: ");
        Serial.print((const char *)gga_info->m_e_w);
        Serial.println((const char *)gga_info->m_lon);
        Serial.print("pos_status: ");
        Serial.print(gga_info->m_pos_status);
        Serial.print(", sat_used: ");
        Serial.println(gga_info->m_sat_used);
        Serial.print("hdop: ");
        Serial.print(gga_info->m_hdop);
        Serial.print(", height: ");
        Serial.print(gga_info->m_height);
        Serial.print(" m, geoid: ");
        Serial.print(gga_info->m_geoid);
        Serial.println(" m");
    }
    else
    {
        // 非測位状態
        // "invalid data"をシリアルモニタへ出力
        Serial.println("invalid data.");
    }
}
void setup_eltres()
{
    // シリアルモニタ出力設定
    Serial.begin(115200);

    // LED初期設定
    pinMode(LED_RUN, OUTPUT);
    digitalWrite(LED_RUN, HIGH);
    pinMode(LED_GNSS, OUTPUT);
    digitalWrite(LED_GNSS, LOW);
    pinMode(LED_SND, OUTPUT);
    digitalWrite(LED_SND, LOW);
    pinMode(LED_ERR, OUTPUT);
    digitalWrite(LED_ERR, LOW);

    // ELTRES起動処理
    eltres_board_result ret = EltresAddonBoard.begin(ELTRES_BOARD_SEND_MODE_1MIN, eltres_event_cb, gga_event_cb);
    if (ret != ELTRES_BOARD_RESULT_OK)
    {
        // ELTRESエラー発生
        digitalWrite(LED_RUN, LOW);
        digitalWrite(LED_ERR, HIGH);
        program_sts = PROGRAM_STS_STOPPED;
        Serial.print("cannot start eltres board (");
        Serial.print(ret);
        Serial.println(").");
    }
    else
    {
        // 正常
        program_sts = PROGRAM_STS_RUNNING;
    }
}
/**
 * @brief GPSペイロード設定
 */
void send_data_eltres(int num_people)
{
    switch (program_sts)
    {
    case PROGRAM_STS_RUNNING:
        // プログラム内部状態：起動中
        if (gnss_recevie_timeout)
        {
            // GNSS電波受信タイムアウト（GNSS受信エラー）時の点滅処理
            uint64_t now_time = millis();
            if ((now_time - last_change_blink_time) >= 1000)
            {
                last_change_blink_time = now_time;
                bool set_value = digitalRead(LED_ERR);
                bool next_value = (set_value == LOW) ? HIGH : LOW;
                digitalWrite(LED_ERR, next_value);
            }
        }
        else
        {
            digitalWrite(LED_ERR, LOW);
        }

        if (event_send_ready)
        {
            // 送信直前通知時の処理
            event_send_ready = false;
            send_data(num_people);
            // 送信ペイロードの設定
            EltresAddonBoard.set_payload(payload);
        }
        break;

    case PROGRAM_STS_STOPPED:
        // プログラム内部状態：終了
        break;
    }
}
void send_data(int num_people)
{
    int spresense_id = 3;
    String lat_string = String((char *)last_gga_info.m_lat);
    String lon_string = String((char *)last_gga_info.m_lon);
    int index;
    uint32_t gnss_time;
    uint32_t utc_time;

    // GNSS時刻(epoch秒)の取得
    EltresAddonBoard.get_gnss_time(&gnss_time);
    // UTC時刻を計算（閏秒補正）
    utc_time = gnss_time - 18;

    // 設定情報をシリアルモニタへ出力
    Serial.print("[setup_payload_gps]");
    Serial.print("lat:");
    Serial.print(lat_string);
    Serial.print(",lon:");
    Serial.print(lon_string);
    Serial.print(",utc:");
    Serial.print(utc_time);
    Serial.print(",pos:");
    Serial.print(last_gga_info.m_pos_status);
    Serial.println();

    // ペイロード領域初期化
    memset(payload, 0x00, sizeof(payload));
    // ペイロード種別[GPSペイロード]設定
    payload[0] = spresense_id;
    payload[1] = num_people;
}
