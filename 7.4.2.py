import datetime

class ContextAgent:
    """
    사용자의 입력을 감지하고, 판단하여, 파일을 수정하는 행동을 하고,
    대화 내용을 학습(기억)하는 간단한 컨텍스트 에이전트 클래스.
    """
    def __init__(self, schedule_file="schedule.txt"):
        """에이전트 초기화: 메모리와 파일 경로 설정"""
        self.memory = []  # 4. 학습: 대화 기록을 저장할 메모리
        self.schedule_file = schedule_file # 3. 행동: 일정을 기록할 파일

    def perceive(self):
        """1. 감지: 사용자로부터 입력을 받는다."""
        print("\n> ", end="")
        user_input = input()
        return user_input

    def decide(self, message: str) -> str:
        """2. 판단: 메시지의 의도를 분류한다."""
        if "일정" in message_lower or "약속" in message_lower:
            return "add_schedule" # 의도: 일정 추가
        elif "종료" in message_lower:
            return "exit" # 의도: 프로그램 종료
        else:
            return "chat" # 의도: 일반 대화

    def act(self, intent: str, message: str) -> str:
        """3. 행동: 판단된 의도에 따라 행동을 수행한다."""
        if intent == "add_schedule":
            # '일정' 의도일 경우, 파일에 내용을 기록하는 행동 수행
            try:
                self._update_schedule_file(message)
                return f"알겠습니다. '{message}' 일정을 파일에 저장했습니다."
            except Exception as e:
                return f"오류가 발생하여 일정을 저장하지 못했습니다: {e}"
        elif intent == "exit":
            return "대화를 종료합니다. 안녕히 가세요."
        else: # 'chat' 의도일 경우
            return "네, 듣고 있어요. 다음 할 일을 말씀해주세요."

    def _update_schedule_file(self, schedule_info: str):
        """
        [행동의 구체적인 구현] schedule.txt 파일에 일정을 추가하는 내부 함수
        """
        # 현재 시간을 YYYY-MM-DD HH:MM:SS 형식으로 기록
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.schedule_file, "a", encoding="utf-8") as f:
            f.write(f"[{now}] {schedule_info}\n")

    def learn(self, user_message: str, agent_response: str):
        """4. 학습: 대화 내용을 메모리에 기록한다."""
        self.memory.append(f"사용자: {user_message}")
        self.memory.append(f"에이전트: {agent_response}")

    def run(self):
        """에이전트의 메인 실행 루프"""
        print("안녕하세요! 저는 당신의 일정 관리 비서입니다. '종료'라고 입력하시면 대화가 끝납니다.")
        while True:
            # 1. 감지
            user_message = self.perceive()
            
            # 2. 판단
            intent = self.decide(user_message)
            
            # 3. 행동
            agent_response = self.act(intent, user_message)
            print(agent_response)
            
            # 4. 학습
            self.learn(user_message, agent_response)
            
            if intent == "exit":
                break

# 에이전트 생성 및 실행
if __name__ == "__main__":
    agent = ContextAgent()
    agent.run()
