on run {phoneNum, msg}
    tell application "Messages"
        set SMSService to 1st account whose service type = SMS
        send msg to buddy phoneNum of SMSService
    end tell
end run