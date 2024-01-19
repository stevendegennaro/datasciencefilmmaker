on run {phoneNum, msg}
	tell application "Messages"
        send msg to buddy phoneNum of service "SMS"
	end tell
end run